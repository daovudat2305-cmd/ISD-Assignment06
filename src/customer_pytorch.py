"""
ISD-Assignment06 | Dataset 2: Hành Vi Khách Hàng
PyTorch Pipeline – Customer Rating Behavior Prediction
=======================================================
Nguồn dữ liệu : data/customer_behavior/
                  ratings.json  → {CustomerID, ProductID, Rate(1–5), CreateDate}
                  products.json → {Id, Name, UnitPrice}
                  customers.json → {Id, NickName}

Mục tiêu      : Dự đoán avg_rating (điểm đánh giá TB) ngày tiếp theo
Mô hình       : Vanilla RNN (PyTorch nn.RNN)

Chiến lược xây dựng chuỗi thời gian:
  - Tổng hợp tất cả ratings theo ngày → time-series hệ thống
  - 18 đặc trưng: rating_count, avg_rating, unique_users, unique_products,
                  high/low ratio, avg_price, std_rating,
                  rolling 7-day stats, lag features,
                  cyclical encoding (day-of-week, month)

Biểu đồ đầu ra (reports/figures/):
  customer_eda_pytorch.png
  customer_rnn_pytorch.png
  customer_comparison_pytorch.png
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
from data_generator import load_customer_behavior_data

# ─────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Customer PyTorch] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# 1. Dataset & Model
# ══════════════════════════════════════════════════════════════
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class CustomerRNNModel(nn.Module):
    """
    Many-to-One Vanilla RNN cho bài toán dự báo hành vi khách hàng.

    Kiến trúc:
      [Input (B, T, 18 features)]
      → [Vanilla RNN (nn.RNN) × 2 layers + Dropout]
      → [Linear(hidden→32) → ReLU → Dropout]
      → [Linear(32→16) → ReLU]
      → [Linear(16→1)]   # dự đoán avg_rating ngày tiếp theo
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, model_type: str = "RNN",
                 dropout: float = 0.2):
        super().__init__()
        _drop = dropout if num_layers > 1 else 0.0
        rnn_map = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = rnn_map[model_type.upper()](
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=_drop
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])   # Many-to-One


# ══════════════════════════════════════════════════════════════
# 2. Training
# ══════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader,
                epochs: int = 60, lr: float = 2e-3, device=DEVICE):
    """Vòng lặp huấn luyện với Gradient Clipping & ReduceLROnPlateau."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=6, min_lr=1e-6
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
# 4. EDA – Phân tích hành vi khách hàng
# ══════════════════════════════════════════════════════════════
def plot_eda(df_agg: pd.DataFrame, df_ratings: pd.DataFrame,
             train_size: int, stats: dict):
    """
    8-panel EDA chart cho Customer Behavior:
      1. Avg rating timeline + rolling 7D + split
      2. Rating count mỗi ngày
      3. Unique users mỗi ngày
      4. Phân phối điểm 1–5 (toàn bộ)
      5. High (≥4) vs Low (≤2) ratio stackplot
      6. Top 10 sản phẩm được đánh giá nhiều nhất
      7. Avg rating theo giờ trong ngày
      8. Heatmap tương quan features
    """
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(20, 28))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    # Panel 1: Avg rating timeline
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df_agg.index, df_agg["avg_rating"], "#80DEEA", lw=1.3, alpha=0.9,
            label="Avg Rating Mỗi Ngày")
    ax.plot(df_agg.index, df_agg["rolling_avg_7d"], "#FF8A65", lw=2.0, ls="--",
            label="Rolling Average 7 Ngày")
    ax.axvline(train_size, color="#69F0AE", lw=2.5, ls=":",
               label=f"Train/Test Split (t={train_size})")
    ax.fill_betweenx([1, 5], 0, train_size, alpha=0.05, color="green", label="Tập Train")
    ax.fill_betweenx([1, 5], train_size, len(df_agg), alpha=0.05, color="red", label="Tập Test")
    ax.set_ylim(1, 5)
    ax.set_title("Điểm Đánh Giá Trung Bình Hàng Ngày & Rolling Average 7 Ngày",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Ngày (Day Index)"); ax.set_ylabel("Avg Rating (1–5)"); ax.legend(fontsize=9)

    # Panel 2: Rating count
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df_agg.index, df_agg["rating_count"], color="#CE93D8", alpha=0.7, width=0.8)
    ax.plot(df_agg.index, df_agg["rolling_count_7d"], "#FF7043", lw=2.0,
            label="Rolling Count 7D")
    ax.set_title("Số Lượng Đánh Giá Mỗi Ngày", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Rating Count"); ax.legend(fontsize=9)

    # Panel 3: Unique users
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df_agg.index, df_agg["unique_users"], "#66BB6A", lw=1.5,
            label="Unique Users")
    ax.fill_between(df_agg.index, 0, df_agg["unique_users"],
                    alpha=0.2, color="#66BB6A")
    ax.set_title("Số Khách Hàng Hoạt Động Mỗi Ngày", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Unique Users")

    # Panel 4: Rating distribution
    ax = fig.add_subplot(gs[2, 0])
    rc = df_ratings["Rate"].value_counts().sort_index()
    bar_colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#26A69A"]
    bars = ax.bar(rc.index, rc.values, color=bar_colors,
                  edgecolor="black", lw=0.5, alpha=0.9)
    for bar, val in zip(bars, rc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Phân Phối Điểm Rating Toàn Bộ (1–5 ⭐)", fontweight="bold")
    ax.set_xlabel("Rating Score"); ax.set_ylabel("Số Lượng Đánh Giá")
    ax.set_xticks([1, 2, 3, 4, 5])

    # Panel 5: High vs Low ratio
    ax = fig.add_subplot(gs[2, 1])
    ax.stackplot(df_agg.index,
                 df_agg["high_ratio"], df_agg["low_ratio"],
                 colors=["#26A69A", "#EF5350"], alpha=0.75,
                 labels=["High Rating (≥4 ⭐)", "Low Rating (≤2 ⭐)"])
    ax.set_title("Tỷ Lệ High vs Low Rating Theo Thời Gian", fontweight="bold")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Tỷ Lệ (Proportion)")
    ax.legend(fontsize=9)

    # Panel 6: Top 10 sản phẩm
    ax = fig.add_subplot(gs[3, 0])
    top10 = df_ratings.groupby("ProductID").size().nlargest(10)
    ax.barh(range(10), top10.values[::-1],
            color="#7E57C2", alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"ProductID {pid}" for pid in top10.index[::-1]], fontsize=9)
    ax.set_title("Top 10 Sản Phẩm Được Đánh Giá Nhiều Nhất", fontweight="bold")
    ax.set_xlabel("Số Lần Được Đánh Giá")

    # Panel 7: Rating theo giờ
    ax = fig.add_subplot(gs[3, 1])
    df_ratings["hour"] = df_ratings["CreateDate"].dt.hour
    hourly_avg = df_ratings.groupby("hour")["Rate"].mean()
    ax.bar(hourly_avg.index, hourly_avg.values, color="#42A5F5",
           alpha=0.75, edgecolor="black", lw=0.3)
    global_avg = df_ratings["Rate"].mean()
    ax.axhline(global_avg, color="red", ls="--", lw=1.5,
               label=f"Global Avg ({global_avg:.2f})")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Điểm Rating Trung Bình Theo Giờ Trong Ngày", fontweight="bold")
    ax.set_xlabel("Giờ (0–23)"); ax.set_ylabel("Avg Rating"); ax.legend(fontsize=9)

    # Panel 8: Correlation heatmap
    ax = fig.add_subplot(gs[4, :])
    feat_cols = ["rating_count", "avg_rating", "unique_users", "unique_products",
                 "high_ratio", "low_ratio", "avg_price", "std_rating",
                 "rolling_avg_7d", "rolling_count_7d"]
    corr = df_agg[feat_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.6})
    ax.set_title("Ma Trận Tương Quan Các Đặc Trưng", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"EDA – Hành Vi Khách Hàng (Customer Rating Behavior) | PyTorch Pipeline\n"
        f"{stats['total_ratings']:,} ratings  |  "
        f"{stats['total_customers']} khách hàng  |  "
        f"{stats['total_products']} sản phẩm  |  "
        f"{stats['date_range']}",
        fontsize=14, fontweight="bold", y=0.995
    )

    path = os.path.join(FIGURES_DIR, "customer_eda_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] EDA → {path}")


# ══════════════════════════════════════════════════════════════
# 5. Visualization kết quả mô hình
# ══════════════════════════════════════════════════════════════
def plot_model_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                       metrics: dict, model_name: str):
    """4-panel: Loss curve + Actual/Pred + Scatter + Residuals."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"PyTorch {model_name} – Dự Báo Avg Rating Hàng Ngày\n"
        f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}",
        fontsize=13, fontweight="bold"
    )
    ep = range(1, len(history["train_loss"]) + 1)

    ax = axes[0, 0]
    ax.plot(ep, history["train_loss"], "#42A5F5", lw=2, label="Train Loss (MSE)")
    ax.plot(ep, history["val_loss"],   "#EF5350", lw=2, ls="--", label="Val Loss (MSE)")
    ax.set_title("MSE Loss Theo Epoch", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(y_true, "#26A69A", lw=1.8, alpha=0.85, label="Thực Tế (Actual)")
    ax.plot(y_pred, "#FF7043", lw=1.6, ls="--", alpha=0.85,
            label=f"Dự Đoán ({model_name})")
    ax.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.12, color="#FF7043")
    ax.set_ylim(1, 5)
    ax.set_title("Avg Rating: Thực Tế vs Dự Đoán (Test Set)", fontweight="bold")
    ax.set_xlabel("Bước Thời Gian (Ngày)"); ax.set_ylabel("Avg Rating (1–5)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#7E57C2", label="Điểm Dự Đoán")
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.8, label="y = x (Lý Tưởng)")
    ax.set_title("Scatter: Thực Tế vs Dự Đoán", fontweight="bold")
    ax.set_xlabel("Avg Rating Thực Tế"); ax.set_ylabel("Avg Rating Dự Đoán")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(residuals, bins=40, color="#66BB6A", alpha=0.75, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.8, label="Zero")
    ax.axvline(residuals.mean(), color="orange", ls=":", lw=2,
               label=f"Mean ({residuals.mean():.3f})")
    ax.set_title("Phân Phối Sai Số (Residuals)", fontweight="bold")
    ax.set_xlabel("Sai Số (Rating)"); ax.set_ylabel("Tần Số")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"customer_{model_name.lower()}_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {model_name} → {path}")


def plot_comparison(results_dict: dict):
    models = list(results_dict.keys())
    palette = ["#42A5F5", "#26A69A", "#FF7043"]
    colors = palette[:len(models)] if len(models) <= len(palette) else palette * (len(models) // len(palette) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("So Sánh Mô Hình PyTorch – Hành Vi Khách Hàng (Avg Rating)",
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
    path = os.path.join(FIGURES_DIR, "customer_comparison_pytorch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Comparison → {path}")


# ══════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ══════════════════════════════════════════════════════════════
def run_pipeline(window_size: int = 10, epochs: int = 60,
                 batch_size: int = 32, hidden_dim: int = 64,
                 train_ratio: float = 0.8, aggregation: str = "daily"):
    """
    Chạy toàn bộ pipeline PyTorch cho dữ liệu hành vi khách hàng:
      1. Load & Aggregate theo ngày
      2. EDA 8-panel
      3. Train Vanilla RNN
      4. Evaluate & Visualize
    """
    print(f"\n{'='*60}")
    print(f"  PYTORCH CUSTOMER PIPELINE (RNN - {aggregation.upper()})")
    print(f"{'='*60}")

    data = load_customer_behavior_data(
        window_size=window_size, train_ratio=train_ratio, aggregation=aggregation
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    target_scaler    = data["target_scaler"]
    input_dim        = X_train.shape[2]

    print("\n[EDA] Đang vẽ biểu đồ phân tích...")
    plot_eda(data["df_agg"], data["df_ratings"], data["train_size"], data["stats"])

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                              batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(X_test,  y_test),
                              batch_size, shuffle=False)

    all_results = {}
    for mtype in ["RNN"]:
        print(f"\n  [Train] {mtype} (Vanilla RNN)...")
        model   = CustomerRNNModel(input_dim, hidden_dim, 2, mtype, 0.2)
        history = train_model(model, train_loader, test_loader, epochs, 2e-3)
        y_true, y_pred = get_predictions(model, test_loader, target_scaler)
        metrics        = compute_metrics(y_true, y_pred)
        all_results[mtype] = metrics
        print(f"  → RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        plot_model_results(history, y_true, y_pred, metrics, mtype)

    if len(all_results) > 1:
        plot_comparison(all_results)

    print(f"\n{'─'*55}")
    print(f"  KẾT QUẢ PYTORCH (RNN) – Hành Vi Khách Hàng")
    print(f"{'─'*55}")
    print(f"  {'Model':<8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'─'*42}")
    for m, r in all_results.items():
        print(f"  {m:<8} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>10.4f}")

    return all_results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = run_pipeline(
        window_size = 10,
        epochs      = 60,
        batch_size  = 32,
        hidden_dim  = 64,
        aggregation = "daily",
    )
