"""
ISD-Assignment06 | Dataset 1: Chứng Khoán
Keras Pipeline – Stock Price Prediction
=========================================
Nguồn dữ liệu : data/stock_price/all_indices_data.csv
Mục tiêu      : Dự đoán giá đóng cửa (Close Price) ngày tiếp theo
Mô hình       : SimpleRNN (Keras / TensorFlow)

Biểu đồ đầu ra (reports/figures/):
  eda_{ticker}_keras.png
  stock_{ticker}_rnn_keras.png
  stock_comparison_{ticker}_keras.png
  stock_framework_{ticker}.png   (Keras vs PyTorch, nếu truyền pytorch_results)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (SimpleRNN, LSTM, GRU,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import load_real_stock_data

# ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
print(f"[Stock Keras] TF {tf.__version__} | "
      f"GPU: {len(tf.config.list_physical_devices('GPU'))}")


# ══════════════════════════════════════════════════════════════
# 1. Model Builder
# ══════════════════════════════════════════════════════════════
def build_model(input_shape: tuple, model_type: str = "RNN",
                units: int = 128, dropout_rate: float = 0.25,
                learning_rate: float = 1e-3):
    """
    Kiến trúc Keras Sequential (Vanilla SimpleRNN):
      [SimpleRNN(units, return_seq=True)] → Dropout → BatchNorm
      [SimpleRNN(units//2, return_seq=False)] → Dropout → BatchNorm
      Dense(64, relu) → Dropout → Dense(32, relu) → Dense(1)

    Parameters
    ----------
    input_shape   : (timesteps, n_features) = (20, 11)
    model_type    : 'RNN', 'LSTM', 'GRU'
    units         : số units tầng 1; tầng 2 = units//2
    dropout_rate  : tỷ lệ dropout
    learning_rate : Adam learning rate
    """
    rnn_map = {"RNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}
    if model_type.upper() not in rnn_map:
        raise ValueError(f"model_type phải là RNN/LSTM/GRU, nhận: {model_type}")
    Layer = rnn_map[model_type.upper()]

    model = Sequential(name=f"{model_type}_stock")
    model.add(Layer(units=units, return_sequences=True,
                    input_shape=input_shape, name=f"{model_type}_1"))
    model.add(Dropout(dropout_rate)); model.add(BatchNormalization())

    model.add(Layer(units=units // 2, return_sequences=False,
                    name=f"{model_type}_2"))
    model.add(Dropout(dropout_rate)); model.add(BatchNormalization())

    model.add(Dense(64, activation="relu")); model.add(Dropout(dropout_rate / 2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, name="output"))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


# ══════════════════════════════════════════════════════════════
# 2. Training
# ══════════════════════════════════════════════════════════════
def train_model(model, X_train, y_train, X_val, y_val,
                epochs: int = 80, batch_size: int = 64, verbose: int = 0):
    """Huấn luyện với EarlyStopping + ReduceLROnPlateau."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                          min_lr=1e-7, verbose=1),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose,
        shuffle=False,   # Time-series: không shuffle!
    )


# ══════════════════════════════════════════════════════════════
# 3. Evaluation
# ══════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ══════════════════════════════════════════════════════════════
# 4. EDA
# ══════════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame, ticker: str, train_size: int):
    """
    6-panel EDA chart (Keras palette):
      1. Giá + MA7/MA21 + split
      2. Volume
      3. Daily Return
      4. RSI 14
      5. Bollinger Width + Volatility (twin-axis)
      6. Heatmap tương quan
    """
    sns.set_theme(style="darkgrid")
    ticker_label = ticker.replace("^", "")
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(df.index, df["close"], "#80DEEA", lw=1.2, alpha=0.9, label="Close")
    ax.plot(df.index, df["ma7"],   "#FFB74D", lw=1.4, label="MA 7")
    ax.plot(df.index, df["ma21"],  "#F06292", lw=1.6, ls="--", label="MA 21")
    ax.axvline(train_size, color="#69F0AE", lw=2.5, ls=":",
               label=f"Train/Test Split (t={train_size})")
    ax.fill_betweenx([df["close"].min(), df["close"].max()],
                      0, train_size, alpha=0.07, color="green", label="Tập Train")
    ax.fill_betweenx([df["close"].min(), df["close"].max()],
                      train_size, len(df), alpha=0.07, color="red", label="Tập Test")
    ax.set_title(f"[{ticker_label}] Giá Đóng Cửa & Đường Trung Bình Động | Keras Pipeline",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("Giá (USD)"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df.index, df["volume"], color="#CE93D8", alpha=0.65, width=1.0)
    ax.set_title("Khối Lượng Giao Dịch (Volume)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("Volume")

    ax = fig.add_subplot(gs[1, 1])
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in df["return"]]
    ax.bar(df.index, df["return"], color=colors, alpha=0.75, width=1.0)
    ax.axhline(0, color="white", lw=0.8)
    ax.set_title("Lợi Nhuận Hàng Ngày (%)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("Return")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df.index, df["rsi_14"], "#FFCA28", lw=1.3, label="RSI 14")
    ax.axhline(70, color="#E53935", ls="--", lw=1.2, label="Quá Mua (70)")
    ax.axhline(30, color="#1E88E5", ls="--", lw=1.2, label="Quá Bán (30)")
    ax.fill_between(df.index, 70, df["rsi_14"].clip(lower=70), alpha=0.3, color="#E53935")
    ax.fill_between(df.index, df["rsi_14"].clip(upper=30), 30, alpha=0.3, color="#1E88E5")
    ax.set_ylim(0, 100)
    ax.set_title("RSI 14 Phiên", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("RSI"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    ax_t = ax.twinx()
    ax.plot(df.index, df["bb_width"], "#4DD0E1", lw=1.3, label="BB Width")
    ax_t.plot(df.index, df["volatility_7d"], "#FFAB40", lw=1.0,
              ls=":", label="Volatility 7D", alpha=0.8)
    ax.set_title("Bollinger Width & Volatility 7D", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("BB Width", color="#4DD0E1")
    ax_t.set_ylabel("Volatility", color="#FFAB40")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax_t.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=9)

    ax = fig.add_subplot(gs[3, :])
    feat = ["open","high","low","close","volume",
            "return","ma7","ma21","volatility_7d","rsi_14","bb_width"]
    corr = df[feat].corr()
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.7})
    ax.set_title("Ma Trận Tương Quan Các Đặc Trưng", fontsize=13, fontweight="bold")

    fig.suptitle(f"EDA – Dữ Liệu Chứng Khoán: {ticker_label} | Keras Pipeline",
                 fontsize=16, fontweight="bold", y=0.99)
    path = os.path.join(FIGURES_DIR, f"eda_{ticker_label}_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] EDA → {path}")


# ══════════════════════════════════════════════════════════════
# 5. Visualization kết quả mô hình
# ══════════════════════════════════════════════════════════════
def plot_model_results(history, y_true: np.ndarray, y_pred: np.ndarray,
                       metrics: dict, model_name: str, ticker: str):
    """4-panel: MSE Loss + MAE + Actual/Pred + Residuals."""
    ticker_label = ticker.replace("^", "")
    residuals    = y_true - y_pred
    hist         = history.history

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Keras {model_name} – Dự Báo Giá Chứng Khoán [{ticker_label}]\n"
        f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    ep = range(1, len(hist["loss"]) + 1)

    ax = axes[0, 0]
    ax.plot(ep, hist["loss"],     "#42A5F5", lw=2, label="Train MSE")
    ax.plot(ep, hist["val_loss"], "#EF5350", lw=2, ls="--", label="Val MSE")
    ax.set_title("Loss (MSE) Theo Epoch", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, hist["mae"],     "#66BB6A", lw=2, label="Train MAE")
    ax.plot(ep, hist["val_mae"], "#FFA726", lw=2, ls="--", label="Val MAE")
    ax.set_title("MAE Theo Epoch", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(y_true, "#26A69A", lw=1.8, alpha=0.85, label="Thực Tế (Actual)")
    ax.plot(y_pred, "#FF7043", lw=1.6, ls="--", alpha=0.85,
            label=f"Dự Đoán Keras {model_name}")
    ax.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.1, color="#FF7043")
    ax.set_title("Giá Đóng Cửa: Thực Tế vs Dự Đoán (Test Set)", fontweight="bold")
    ax.set_xlabel("Bước Thời Gian (Phiên)"); ax.set_ylabel("Giá (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(residuals, bins=40, color="#AB47BC", alpha=0.75, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.8, label="Zero")
    ax.axvline(residuals.mean(), color="yellow", ls=":", lw=2,
               label=f"Mean ({residuals.mean():.2f})")
    ax.set_title("Phân Phối Sai Số (Residuals)", fontweight="bold")
    ax.set_xlabel("Sai Số (USD)"); ax.set_ylabel("Tần Số")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"stock_{ticker_label}_{model_name.lower()}_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {model_name} → {path}")


def plot_comparison(results_dict: dict, ticker: str):
    ticker_label = ticker.replace("^", "")
    models = list(results_dict.keys())
    palette = ["#FF7043", "#FFA726", "#26A69A"]
    colors = palette[:len(models)] if len(models) <= len(palette) else palette * (len(models) // len(palette) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"So Sánh Mô Hình Keras – Chứng Khoán [{ticker_label}]",
                 fontsize=14, fontweight="bold")

    for ax, metric, title, best_fn in [
        (axes[0], "rmse", "RMSE (↓ tốt hơn)", min),
        (axes[1], "mae",  "MAE  (↓ tốt hơn)", min),
        (axes[2], "r2",   "R² Score (↑ tốt hơn)", max),
    ]:
        vals = [results_dict[m][metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="black", lw=0.8, alpha=0.85)
        best_v = best_fn(vals)
        for bar, val in zip(bars, vals):
            if abs(val - best_v) < 1e-9:
                bar.set_edgecolor("gold"); bar.set_linewidth(3)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric.upper()); ax.set_xlabel("Mô Hình")
        ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"stock_comparison_{ticker_label}_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Comparison → {path}")


def plot_framework_comparison(keras_results: dict, pytorch_results: dict, ticker: str):
    """Grouped bar: PyTorch vs Keras cho các mô hình (mặc định Vanilla RNN)."""
    ticker_label = ticker.replace("^", "")
    model_names  = [m for m in ["RNN", "LSTM", "GRU"] if m in keras_results or m in pytorch_results]
    if not model_names:
        model_names = ["RNN"]
    x = np.arange(len(model_names)); width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"So Sánh Framework: PyTorch vs Keras (Vanilla RNN) – Chứng Khoán [{ticker_label}]",
                 fontsize=14, fontweight="bold")

    for ax, metric, title in [
        (axes[0], "rmse", "RMSE (↓)"),
        (axes[1], "mae",  "MAE  (↓)"),
        (axes[2], "r2",   "R² Score (↑)"),
    ]:
        p_vals = [pytorch_results.get(m, {}).get(metric, 0) for m in model_names]
        k_vals = [keras_results.get(m,   {}).get(metric, 0) for m in model_names]
        b1 = ax.bar(x - width/2, p_vals, width, label="PyTorch",
                    color="#42A5F5", alpha=0.85, edgecolor="black", lw=0.7)
        b2 = ax.bar(x + width/2, k_vals, width, label="Keras",
                    color="#FF7043", alpha=0.85, edgecolor="black", lw=0.7)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(model_names)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Mô Hình"); ax.legend(); ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"stock_framework_{ticker_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Framework comparison → {path}")


# ══════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ══════════════════════════════════════════════════════════════
def run_pipeline(ticker: str = "^GSPC", window_size: int = 20,
                 epochs: int = 80, batch_size: int = 64,
                 units: int = 128, train_ratio: float = 0.8,
                 pytorch_results: dict = None):
    """
    Chạy toàn bộ pipeline Keras cho dữ liệu chứng khoán:
      1. Load & EDA
      2. Train SimpleRNN
      3. Evaluate & Visualize
      4. (Tùy chọn) So sánh với PyTorch
    """
    ticker_label = ticker.replace("^", "")
    print(f"\n{'='*60}")
    print(f"  KERAS STOCK PIPELINE (RNN) – {ticker_label}")
    print(f"{'='*60}")

    data = load_real_stock_data(ticker=ticker, window_size=window_size,
                                train_ratio=train_ratio)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    target_scaler    = data["target_scaler"]
    input_shape      = (X_train.shape[1], X_train.shape[2])

    print("\n[EDA] Đang vẽ biểu đồ phân tích...")
    plot_eda(data["df_full"], ticker, data["train_size"])

    all_results = {}
    for mtype in ["RNN"]:
        print(f"\n  [Train] Keras {mtype} (SimpleRNN)...")
        model = build_model(input_shape, mtype, units, 0.25, 1e-3)
        model.summary()

        history = train_model(model, X_train, y_train, X_test, y_test,
                              epochs, batch_size, verbose=0)

        y_pred = target_scaler.inverse_transform(
            model.predict(X_test, verbose=0)).flatten()
        y_true = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_true, y_pred)
        all_results[mtype] = metrics
        print(f"  → RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        plot_model_results(history, y_true, y_pred, metrics, mtype, ticker)

    if len(all_results) > 1:
        plot_comparison(all_results, ticker)

    if pytorch_results is not None:
        plot_framework_comparison(all_results, pytorch_results, ticker)

    print(f"\n{'─'*55}")
    print(f"  KẾT QUẢ KERAS (RNN) – Chứng Khoán [{ticker_label}]")
    print(f"{'─'*55}")
    print(f"  {'Model':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'─'*42}")
    for m, r in all_results.items():
        print(f"  {m:<8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f}")

    return all_results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    keras_res = run_pipeline(
        ticker     = "^GSPC",
        window_size= 20,
        epochs     = 80,
        batch_size = 64,
        units      = 128,
    )
