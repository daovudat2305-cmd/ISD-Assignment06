"""
ISD-Assignment06 | Dataset 1: Chứng Khoán
PyTorch Pipeline – Stock Price Prediction
==========================================
Nguồn dữ liệu : data/stock_price/all_indices_data.csv
Mục tiêu      : Dự đoán giá đóng cửa (Close Price) ngày tiếp theo
Mô hình       : Vanilla RNN (PyTorch nn.RNN)

Biểu đồ đầu ra (reports/figures/):
  eda_{ticker}_pytorch.png
  stock_{ticker}_rnn_pytorch.png
  stock_comparison_{ticker}_pytorch.png
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import load_real_stock_data

# ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stock PyTorch] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# 1. Dataset & Model
# ══════════════════════════════════════════════════════════════
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class StockRNNModel(nn.Module):
    """
    Many-to-One Vanilla RNN cho bài toán dự báo giá chứng khoán.

    Kiến trúc:
      [Input (B, T, 11 features)]
      → [Vanilla RNN (nn.RNN) × 2 layers + Dropout]
      → [Linear(hidden→64) → ReLU → Dropout]
      → [Linear(64→32) → ReLU]
      → [Linear(32→1)]   # dự đoán close price (đã chuẩn hóa)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, model_type: str = "RNN",
                 dropout: float = 0.25):
        super().__init__()
        _drop = dropout if num_layers > 1 else 0.0
        rnn_map = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = rnn_map[model_type.upper()](
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=_drop
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])   # Many-to-One


# ══════════════════════════════════════════════════════════════
# 2. Training
# ══════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader,
                epochs: int = 60, lr: float = 1e-3, device=DEVICE):
    """Vòng lặp huấn luyện với Gradient Clipping & ReduceLROnPlateau."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=7, min_lr=1e-6
    )
    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                v_losses.append(criterion(model(Xv.to(device)), yv.to(device)).item())

        avg_t = float(np.mean(t_losses))
        avg_v = float(np.mean(v_losses))
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        scheduler.step(avg_v)

        if avg_v < best_val:
            best_val   = avg_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{epoch:03d}/{epochs}] Train: {avg_t:.6f} | Val: {avg_v:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return history


# ══════════════════════════════════════════════════════════════
# 3. Evaluation
# ══════════════════════════════════════════════════════════════
def get_predictions(model, loader, target_scaler, device=DEVICE):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(model(Xb.to(device)).cpu().numpy())
            trues.extend(yb.numpy())
    y_pred = target_scaler.inverse_transform(np.array(preds)).flatten()
    y_true = target_scaler.inverse_transform(np.array(trues)).flatten()
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ══════════════════════════════════════════════════════════════
# 4. EDA – Phân tích dữ liệu chứng khoán
# ══════════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame, ticker: str, train_size: int):
    """
    6-panel EDA chart:
      1. Giá + MA7/MA21 + Train/Test split
      2. Khối lượng giao dịch (Volume)
      3. Lợi nhuận hàng ngày (Daily Return)
      4. RSI 14 phiên
      5. Bollinger Band Width
      6. Heatmap tương quan features
    """
    sns.set_theme(style="darkgrid")
    ticker_label = ticker.replace("^", "")
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Giá + MA + split
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df.index, df["close"], "#4FC3F7", lw=1.2, alpha=0.9, label="Close")
    ax.plot(df.index, df["ma7"],   "#FFB74D", lw=1.4, label="MA 7")
    ax.plot(df.index, df["ma21"],  "#EF5350", lw=1.6, ls="--", label="MA 21")
    ax.axvline(train_size, color="#66BB6A", lw=2.5, ls=":",
               label=f"Train/Test Split (t={train_size})")
    ax.fill_betweenx([df["close"].min(), df["close"].max()],
                      0, train_size, alpha=0.06, color="green", label="Tập Train")
    ax.fill_betweenx([df["close"].min(), df["close"].max()],
                      train_size, len(df), alpha=0.06, color="red", label="Tập Test")
    ax.set_title(f"[{ticker_label}] Giá Đóng Cửa Lịch Sử & Đường Trung Bình Động",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Số Phiên Giao Dịch"); ax.set_ylabel("Giá (USD)"); ax.legend(fontsize=9)

    # Panel 2: Volume
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df.index, df["volume"], color="#AB47BC", alpha=0.6, width=1.0)
    ax.set_title("Khối Lượng Giao Dịch (Volume)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("Volume")

    # Panel 3: Daily Return
    ax = fig.add_subplot(gs[1, 1])
    pos = df["return"].clip(lower=0); neg = df["return"].clip(upper=0)
    ax.bar(df.index, pos, color="#26A69A", alpha=0.8, width=1.0, label="Lợi nhuận (+)")
    ax.bar(df.index, neg, color="#EF5350", alpha=0.8, width=1.0, label="Lỗ (-)")
    ax.axhline(0, color="white", lw=0.8)
    ax.set_title("Lợi Nhuận Hàng Ngày (Daily Return)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("Return"); ax.legend(fontsize=9)

    # Panel 4: RSI
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df.index, df["rsi_14"], "#FFA726", lw=1.3, label="RSI 14")
    ax.axhline(70, color="#EF5350", ls="--", lw=1.2, label="Quá Mua (70)")
    ax.axhline(30, color="#42A5F5", ls="--", lw=1.2, label="Quá Bán (30)")
    ax.fill_between(df.index, 70, df["rsi_14"].clip(lower=70), alpha=0.3, color="#EF5350")
    ax.fill_between(df.index, df["rsi_14"].clip(upper=30), 30, alpha=0.3, color="#42A5F5")
    ax.set_ylim(0, 100)
    ax.set_title("RSI 14 Phiên (Relative Strength Index)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("RSI"); ax.legend(fontsize=9)

    # Panel 5: Bollinger
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(df.index, df["bb_width"], "#26C6DA", lw=1.3)
    ax.fill_between(df.index, 0, df["bb_width"], alpha=0.25, color="#26C6DA")
    ax.set_title("Bollinger Band Width (Volatility Proxy)", fontweight="bold")
    ax.set_xlabel("Số Phiên"); ax.set_ylabel("BB Width")

    # Panel 6: Correlation heatmap
    ax = fig.add_subplot(gs[3, :])
    feat_cols = ["open","high","low","close","volume",
                 "return","ma7","ma21","volatility_7d","rsi_14","bb_width"]
    corr = df[feat_cols].corr()
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.7})
    ax.set_title("Ma Trận Tương Quan Các Đặc Trưng", fontsize=13, fontweight="bold")

    fig.suptitle(f"EDA – Dữ Liệu Chứng Khoán: {ticker_label} | PyTorch Pipeline",
                 fontsize=16, fontweight="bold", y=0.99)

    path = os.path.join(FIGURES_DIR, f"eda_{ticker_label}_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] EDA → {path}")


# ══════════════════════════════════════════════════════════════
# 5. Visualization kết quả mô hình
# ══════════════════════════════════════════════════════════════
def plot_model_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                       metrics: dict, model_name: str, ticker: str):
    """4-panel: Loss curve + Actual/Pred + Scatter + Residuals."""
    ticker_label = ticker.replace("^", "")
    residuals    = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"PyTorch {model_name} – Dự Báo Giá Chứng Khoán [{ticker_label}]\n"
        f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )

    ep = range(1, len(history["train_loss"]) + 1)

    ax = axes[0, 0]
    ax.plot(ep, history["train_loss"], "#42A5F5", lw=2, label="Train Loss")
    ax.plot(ep, history["val_loss"],   "#EF5350", lw=2, ls="--", label="Val Loss")
    ax.set_title("MSE Loss Theo Epoch", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(y_true, "#26A69A", lw=1.8, alpha=0.85, label="Thực Tế (Actual)")
    ax.plot(y_pred, "#FF7043", lw=1.6, ls="--", alpha=0.85,
            label=f"Dự Đoán ({model_name})")
    ax.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.12, color="#FF7043")
    ax.set_title("Giá Đóng Cửa: Thực Tế vs Dự Đoán (Test Set)", fontweight="bold")
    ax.set_xlabel("Bước Thời Gian (Phiên)"); ax.set_ylabel("Giá (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(y_true, y_pred, alpha=0.45, s=12, color="#7E57C2", label="Điểm Dự Đoán")
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.8, label="y = x (Lý Tưởng)")
    ax.set_title("Scatter: Thực Tế vs Dự Đoán", fontweight="bold")
    ax.set_xlabel("Giá Thực Tế (USD)"); ax.set_ylabel("Giá Dự Đoán (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(residuals, bins=40, color="#66BB6A", alpha=0.75, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.8, label="Zero")
    ax.axvline(residuals.mean(), color="orange", ls=":", lw=2,
               label=f"Mean ({residuals.mean():.2f})")
    ax.set_title("Phân Phối Sai Số (Residuals = Actual – Predicted)", fontweight="bold")
    ax.set_xlabel("Sai Số (USD)"); ax.set_ylabel("Tần Số")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"stock_{ticker_label}_{model_name.lower()}_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {model_name} → {path}")


def plot_comparison(results_dict: dict, ticker: str):
    """Bar chart so sánh RMSE / MAE / R² của các mô hình."""
    ticker_label = ticker.replace("^", "")
    models = list(results_dict.keys())
    palette = ["#42A5F5", "#26A69A", "#FF7043"]
    colors = palette[:len(models)] if len(models) <= len(palette) else palette * (len(models) // len(palette) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"So Sánh Mô Hình PyTorch – Chứng Khoán [{ticker_label}]",
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
    path = os.path.join(FIGURES_DIR, f"stock_comparison_{ticker_label}_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Comparison → {path}")


# ══════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ══════════════════════════════════════════════════════════════
def run_pipeline(ticker: str = "^GSPC", window_size: int = 20,
                 epochs: int = 60, batch_size: int = 64,
                 hidden_dim: int = 128, train_ratio: float = 0.8):
    """
    Chạy toàn bộ pipeline PyTorch cho dữ liệu chứng khoán:
      1. Load & EDA
      2. Train Vanilla RNN
      3. Evaluate & Visualize
    """
    ticker_label = ticker.replace("^", "")
    print(f"\n{'='*60}")
    print(f"  PYTORCH STOCK PIPELINE (RNN) – {ticker_label}")
    print(f"{'='*60}")

    # ── Load data ──
    data = load_real_stock_data(ticker=ticker, window_size=window_size,
                                train_ratio=train_ratio)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    target_scaler    = data["target_scaler"]
    input_dim        = X_train.shape[2]

    # ── EDA ──
    print("\n[EDA] Đang vẽ biểu đồ phân tích...")
    plot_eda(data["df_full"], ticker, data["train_size"])

    # ── DataLoaders ──
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                              batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(X_test,  y_test),
                              batch_size, shuffle=False)

    # ── Train Vanilla RNN ──
    all_results = {}
    for mtype in ["RNN"]:
        print(f"\n  [Train] {mtype} (Vanilla RNN)...")
        model   = StockRNNModel(input_dim, hidden_dim, 2, mtype, 0.25)
        history = train_model(model, train_loader, test_loader, epochs, 1e-3)
        y_true, y_pred = get_predictions(model, test_loader, target_scaler)
        metrics        = compute_metrics(y_true, y_pred)
        all_results[mtype] = metrics
        print(f"  → RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        plot_model_results(history, y_true, y_pred, metrics, mtype, ticker)

    # ── So sánh (nếu nhiều mô hình) ──
    if len(all_results) > 1:
        plot_comparison(all_results, ticker)

    # ── Tóm tắt ──
    print(f"\n{'─'*55}")
    print(f"  KẾT QUẢ PYTORCH (RNN) – Chứng Khoán [{ticker_label}]")
    print(f"{'─'*55}")
    print(f"  {'Model':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'─'*42}")
    for m, r in all_results.items():
        print(f"  {m:<8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f}")

    return all_results


# ══════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = run_pipeline(
        ticker     = "^GSPC",   # S&P 500
        window_size= 20,
        epochs     = 60,
        batch_size = 64,
        hidden_dim = 128,
    )
    # Thêm ticker khác (tùy chọn):
    # run_pipeline(ticker="^DJI",   epochs=50)
    # run_pipeline(ticker="^N225",  epochs=50)
    # run_pipeline(ticker="^IXIC",  epochs=50)
