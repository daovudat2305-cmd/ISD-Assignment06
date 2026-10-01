"""
ISD-Assignment06 | Dataset 2: Hành Vi Khách Hàng
Keras Pipeline – Customer Rating Behavior Prediction
=====================================================
Nguồn dữ liệu : data/customer_behavior/
                  ratings.json  → {CustomerID, ProductID, Rate(1–5), CreateDate}
                  products.json → {Id, Name, UnitPrice}
                  customers.json → {Id, NickName}

Mục tiêu      : Dự đoán avg_rating (điểm đánh giá TB) ngày tiếp theo
Mô hình       : SimpleRNN (Keras / TensorFlow)

Biểu đồ đầu ra (reports/figures/):
  customer_eda_keras.png
  customer_rnn_keras.png
  customer_comparison_keras.png
  customer_framework.png   (Keras vs PyTorch, nếu truyền pytorch_results)
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
from data_generator import load_customer_behavior_data

# ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
print(f"[Customer Keras] TF {tf.__version__} | "
      f"GPU: {len(tf.config.list_physical_devices('GPU'))}")


# ══════════════════════════════════════════════════════════════
# 1. Model Builder
# ══════════════════════════════════════════════════════════════
def build_model(input_shape: tuple, model_type: str = "RNN",
                units: int = 64, dropout_rate: float = 0.2,
                learning_rate: float = 2e-3):
    """
    Kiến trúc Keras Sequential (Vanilla SimpleRNN):
      [SimpleRNN(units, return_seq=True)] → Dropout → BatchNorm
      [SimpleRNN(units//2, return_seq=False)] → Dropout → BatchNorm
      Dense(32, relu) → Dropout → Dense(16, relu) → Dense(1)

    Parameters
    ----------
    input_shape   : (timesteps, n_features) = (10, 18)
    model_type    : 'RNN', 'LSTM', 'GRU'
    units         : số units tầng 1; tầng 2 = units//2
    dropout_rate  : tỷ lệ dropout
    learning_rate : Adam learning rate
    """
    rnn_map = {"RNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}
    if model_type.upper() not in rnn_map:
        raise ValueError(f"model_type phải là RNN/LSTM/GRU, nhận: {model_type}")
    Layer = rnn_map[model_type.upper()]

    model = Sequential(name=f"{model_type}_customer")
    model.add(Layer(units=units, return_sequences=True,
                    input_shape=input_shape, name=f"{model_type}_1"))
    model.add(Dropout(dropout_rate)); model.add(BatchNormalization())

    model.add(Layer(units=units // 2, return_sequences=False,
                    name=f"{model_type}_2"))
    model.add(Dropout(dropout_rate)); model.add(BatchNormalization())

    model.add(Dense(32, activation="relu")); model.add(Dropout(dropout_rate / 2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, name="output"))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


# ══════════════════════════════════════════════════════════════
# 2. Training
# ══════════════════════════════════════════════════════════════
def train_model(model, X_train, y_train, X_val, y_val,
                epochs: int = 80, batch_size: int = 32, verbose: int = 0):
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
def plot_eda(df_agg: pd.DataFrame, df_ratings: pd.DataFrame,
             train_size: int, stats: dict):
    """
    8-panel EDA (Keras palette):
      1. Avg rating timeline
      2. Rating count + rolling 7D
      3. Unique users
      4. Phân phối 1–5 ⭐
      5. High/Low ratio stackplot
      6. Top 10 sản phẩm
      7. Avg rating theo giờ
      8. Heatmap tương quan
    """
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(20, 28))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(df_agg.index, df_agg["avg_rating"], "#80CBC4", lw=1.3, alpha=0.9,
            label="Avg Rating Mỗi Ngày")
    ax.plot(df_agg.index, df_agg["rolling_avg_7d"], "#FF8A65", lw=2.0, ls="--",
            label="Rolling Average 7 Ngày")
    ax.axvline(train_size, color="#B2FF59", lw=2.5, ls=":",
               label=f"Train/Test Split (t={train_size})")
    ax.fill_betweenx([1, 5], 0, train_size, alpha=0.05, color="green", label="Tập Train")
    ax.fill_betweenx([1, 5], train_size, len(df_agg), alpha=0.05, color="red", label="Tập Test")
    ax.set_ylim(1, 5)
    ax.set_title("Điểm Đánh Giá Trung Bình Hàng Ngày | Keras Pipeline",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Ngày (Day Index)"); ax.set_ylabel("Avg Rating (1–5)"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df_agg.index, df_agg["rating_count"], color="#CE93D8", alpha=0.7, width=0.8)
    ax.plot(df_agg.index, df_agg["rolling_count_7d"], "#FF7043", lw=2.0,
            label="Rolling Count 7D")
    ax.set_title("Số Lượng Đánh Giá Mỗi Ngày", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Rating Count"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df_agg.index, df_agg["unique_users"], "#4DB6AC", lw=1.5)
    ax.fill_between(df_agg.index, 0, df_agg["unique_users"],
                    alpha=0.2, color="#4DB6AC")
    ax.set_title("Số Khách Hàng Hoạt Động Mỗi Ngày", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Unique Users")

    ax = fig.add_subplot(gs[2, 0])
    rc = df_ratings["Rate"].value_counts().sort_index()
    ax.bar(rc.index, rc.values,
           color=["#EF5350","#FF7043","#FFA726","#66BB6A","#26A69A"],
           edgecolor="black", lw=0.5, alpha=0.9)
    for x_pos, v in zip(rc.index, rc.values):
        ax.text(x_pos, v * 1.01, f"{v:,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Phân Phối Điểm Rating (1–5 ⭐)", fontweight="bold")
    ax.set_xlabel("Rating Score"); ax.set_ylabel("Số Lượng")
    ax.set_xticks([1, 2, 3, 4, 5])

    ax = fig.add_subplot(gs[2, 1])
    ax.stackplot(df_agg.index,
                 df_agg["high_ratio"], df_agg["low_ratio"],
                 colors=["#26A69A","#EF5350"], alpha=0.75,
                 labels=["High (≥4 ⭐)", "Low (≤2 ⭐)"])
    ax.set_title("Tỷ Lệ High vs Low Rating Theo Thời Gian", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Tỷ Lệ"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[3, 0])
    top10 = df_ratings.groupby("ProductID").size().nlargest(10)
    ax.barh(range(10), top10.values[::-1],
            color="#7E57C2", alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"ProductID {p}" for p in top10.index[::-1]], fontsize=9)
    ax.set_title("Top 10 Sản Phẩm Được Đánh Giá Nhiều Nhất", fontweight="bold")
    ax.set_xlabel("Số Lần")

    ax = fig.add_subplot(gs[3, 1])
    df_ratings["hour"] = df_ratings["CreateDate"].dt.hour
    hourly_avg = df_ratings.groupby("hour")["Rate"].mean()
    ax.bar(hourly_avg.index, hourly_avg.values, color="#42A5F5",
           alpha=0.75, edgecolor="black", lw=0.3)
    global_avg = df_ratings["Rate"].mean()
    ax.axhline(global_avg, color="red", ls="--", lw=1.5,
               label=f"Global ({global_avg:.2f})")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Avg Rating Theo Giờ Trong Ngày", fontweight="bold")
    ax.set_xlabel("Giờ"); ax.set_ylabel("Avg Rating"); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[4, :])
    feat_cols = ["rating_count","avg_rating","unique_users","unique_products",
                 "high_ratio","low_ratio","avg_price","std_rating",
                 "rolling_avg_7d","rolling_count_7d"]
    corr = df_agg[feat_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.6})
    ax.set_title("Ma Trận Tương Quan Các Đặc Trưng", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"EDA – Hành Vi Khách Hàng (Customer Rating Behavior) | Keras Pipeline\n"
        f"{stats['total_ratings']:,} ratings  |  "
        f"{stats['total_customers']} khách hàng  |  "
        f"{stats['total_products']} sản phẩm  |  "
        f"{stats['date_range']}",
        fontsize=13, fontweight="bold", y=0.995
    )

    path = os.path.join(FIGURES_DIR, "customer_eda_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] EDA → {path}")


# ══════════════════════════════════════════════════════════════
# 5. Visualization kết quả mô hình
# ══════════════════════════════════════════════════════════════
def plot_model_results(history, y_true: np.ndarray, y_pred: np.ndarray,
                       metrics: dict, model_name: str):
    """4-panel: MSE Loss + MAE + Actual/Pred + Residuals."""
    residuals = y_true - y_pred
    hist      = history.history

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Keras {model_name} – Dự Báo Avg Rating Hàng Ngày\n"
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
    ax.set_ylim(1, 5)
    ax.set_title("Avg Rating: Thực Tế vs Dự Đoán (Test Set)", fontweight="bold")
    ax.set_xlabel("Bước Thời Gian (Ngày)"); ax.set_ylabel("Avg Rating (1–5)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(residuals, bins=40, color="#AB47BC", alpha=0.75, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.8, label="Zero")
    ax.axvline(residuals.mean(), color="yellow", ls=":", lw=2,
               label=f"Mean ({residuals.mean():.3f})")
    ax.set_title("Phân Phối Sai Số (Residuals)", fontweight="bold")
    ax.set_xlabel("Sai Số (Rating)"); ax.set_ylabel("Tần Số")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"customer_{model_name.lower()}_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {model_name} → {path}")


def plot_comparison(results_dict: dict):
    models = list(results_dict.keys())
    palette = ["#FF7043", "#FFA726", "#26A69A"]
    colors = palette[:len(models)] if len(models) <= len(palette) else palette * (len(models) // len(palette) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("So Sánh Mô Hình Keras – Hành Vi Khách Hàng (Avg Rating)",
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
    path = os.path.join(FIGURES_DIR, "customer_comparison_keras.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Comparison → {path}")


def plot_framework_comparison(keras_results: dict, pytorch_results: dict):
    """Grouped bar: PyTorch vs Keras cho các mô hình (mặc định Vanilla RNN)."""
    model_names = [m for m in ["RNN", "LSTM", "GRU"] if m in keras_results or m in pytorch_results]
    if not model_names:
        model_names = ["RNN"]
    x = np.arange(len(model_names)); width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("So Sánh Framework: PyTorch vs Keras (Vanilla RNN) – Hành Vi Khách Hàng",
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
    path = os.path.join(FIGURES_DIR, "customer_framework.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Framework comparison → {path}")


# ══════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ══════════════════════════════════════════════════════════════
def run_pipeline(window_size: int = 10, epochs: int = 80,
                 batch_size: int = 32, units: int = 64,
                 train_ratio: float = 0.8, aggregation: str = "daily",
                 pytorch_results: dict = None):
    """
    Chạy toàn bộ pipeline Keras cho dữ liệu hành vi khách hàng:
      1. Load & Aggregate theo ngày
      2. EDA 8-panel
      3. Train SimpleRNN
      4. Evaluate & Visualize
      5. (Tùy chọn) So sánh với PyTorch
    """
    print(f"\n{'='*60}")
    print(f"  KERAS CUSTOMER PIPELINE (RNN - {aggregation.upper()})")
    print(f"{'='*60}")

    data = load_customer_behavior_data(
        window_size=window_size, train_ratio=train_ratio, aggregation=aggregation
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    target_scaler    = data["target_scaler"]
    input_shape      = (X_train.shape[1], X_train.shape[2])

    print("\n[EDA] Đang vẽ biểu đồ phân tích...")
    plot_eda(data["df_agg"], data["df_ratings"], data["train_size"], data["stats"])

    all_results = {}
    for mtype in ["RNN"]:
        print(f"\n  [Train] Keras {mtype} (SimpleRNN)...")
        model = build_model(input_shape, mtype, units, 0.2, 2e-3)
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
        plot_model_results(history, y_true, y_pred, metrics, mtype)

    if len(all_results) > 1:
        plot_comparison(all_results)

    if pytorch_results is not None:
        plot_framework_comparison(all_results, pytorch_results)

    print(f"\n{'─'*55}")
    print(f"  KẾT QUẢ KERAS (RNN) – Hành Vi Khách Hàng")
    print(f"{'─'*55}")
    print(f"  {'Model':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'─'*42}")
    for m, r in all_results.items():
        print(f"  {m:<8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f}")

    return all_results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    keras_res = run_pipeline(
        window_size = 10,
        epochs      = 80,
        batch_size  = 32,
        units       = 64,
        aggregation = "daily",
    )
