"""
ISD-Assignment06: Data Generator / Loader
==========================================
Hỗ trợ 2 nguồn dữ liệu THỰC TẾ:
  1. Chứng khoán  – data/stock_price/all_indices_data.csv
                   (columns: date, open, high, low, close, volume, ticker)
  2. Hành vi khách hàng – data/customer_behavior/
                          ratings.json, products.json, customers.json
                          (CustomerID, ProductID, Rate, CreateDate)

Tiền xử lý chung:
  - Chia train/test theo trục thời gian (Time-based split – tránh Data Leakage)
  - Chuẩn hóa MinMaxScaler (chỉ fit trên tập train)
  - Tạo Sliding Window tensor 3D (samples, timesteps, features)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────
# 0. Đường dẫn mặc định
# ─────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

# Chứng khoán
ALL_INDICES_CSV = os.path.join(_BASE_DIR, "data", "stock_price", "all_indices_data.csv")
INDIVIDUAL_DIR  = os.path.join(_BASE_DIR, "data", "stock_price", "individual_indices_data")

# Hành vi khách hàng
CUSTOMER_DIR    = os.path.join(_BASE_DIR, "data", "customer_behavior")
RATINGS_JSON    = os.path.join(CUSTOMER_DIR, "ratings.json")
PRODUCTS_JSON   = os.path.join(CUSTOMER_DIR, "products.json")
CUSTOMERS_JSON  = os.path.join(CUSTOMER_DIR, "customers.json")


# ─────────────────────────────────────────────────────────────
# 1. Tiện ích: Tạo Sliding Window 3D tensor
# ─────────────────────────────────────────────────────────────
def create_sliding_windows(data: np.ndarray,
                           target_col_idx: int,
                           window_size: int = 20):
    """
    Tạo cặp (X, y) dạng sliding window:
      X[i] = data[i : i+window_size, :]          → (window_size, n_features)
      y[i] = data[i+window_size, target_col_idx]  → scalar
    Trả về: X (N, W, F), y (N,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2. Tiện ích: Thêm chỉ báo kỹ thuật cho chứng khoán
# ─────────────────────────────────────────────────────────────
def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm Return, MA7, MA21, Volatility_7D, RSI_14, BB_Width từ OHLCV."""
    df = df.copy()
    df["return"]        = df["close"].pct_change()
    df["ma7"]           = df["close"].rolling(7).mean()
    df["ma21"]          = df["close"].rolling(21).mean()
    df["volatility_7d"] = df["return"].rolling(7).std()

    delta   = df["close"].diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    std21        = df["close"].rolling(21).std()
    df["bb_width"] = (2 * std21) / (df["ma21"] + 1e-9)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 3. Load dữ liệu chứng khoán THỰC TẾ
# ─────────────────────────────────────────────────────────────
def load_real_stock_data(ticker: str = "^GSPC",
                         window_size: int = 20,
                         train_ratio: float = 0.8,
                         min_rows: int = 500):
    """
    Tải & tiền xử lý dữ liệu chứng khoán từ all_indices_data.csv.

    Returns dict:
      X_train, y_train, X_test, y_test, target_scaler,
      feature_names, df_full, train_size, ticker
    """
    if not os.path.exists(ALL_INDICES_CSV):
        raise FileNotFoundError(f"Không tìm thấy: {ALL_INDICES_CSV}")

    df_all = pd.read_csv(ALL_INDICES_CSV, parse_dates=["date"])
    df_all.columns = [c.strip().lower() for c in df_all.columns]

    df = df_all[df_all["ticker"] == ticker].copy()
    if len(df) < min_rows:
        individual_path = os.path.join(INDIVIDUAL_DIR, f"{ticker}_data.csv")
        if os.path.exists(individual_path):
            df = pd.read_csv(individual_path, parse_dates=["date"])
            df.columns = [c.strip().lower() for c in df.columns]
        else:
            raise ValueError(f"Ticker '{ticker}' không đủ dữ liệu.")

    df.sort_values("date", inplace=True)
    df = df[df["close"] > 0].dropna(subset=["open","high","low","close","volume"])
    df.reset_index(drop=True, inplace=True)
    df = _add_technical_indicators(df)

    feature_names = ["open","high","low","close","volume",
                     "return","ma7","ma21","volatility_7d","rsi_14","bb_width"]
    target_idx    = feature_names.index("close")
    data          = df[feature_names].values

    train_size  = int(len(data) * train_ratio)
    train_raw   = data[:train_size]
    test_raw    = data[train_size:]

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    X_train, y_train = create_sliding_windows(train_scaled, target_idx, window_size)
    X_test,  y_test  = create_sliding_windows(test_scaled,  target_idx, window_size)

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_raw[:, target_idx : target_idx + 1])

    print(f"[Stock] Ticker: {ticker} | Tổng: {len(df):,} phiên | "
          f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"Features: {len(feature_names)} | Window: {window_size}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "df_full":       df,
        "train_size":    train_size,
        "ticker":        ticker,
    }


# ─────────────────────────────────────────────────────────────
# 4. Load dữ liệu Hành vi Khách hàng THỰC TẾ
# ─────────────────────────────────────────────────────────────
def load_customer_behavior_data(window_size: int = 10,
                                train_ratio: float = 0.8,
                                min_ratings_per_user: int = 15,
                                aggregation: str = "daily",
                                top_n_users: int = None):
    """
    Tải & tiền xử lý dữ liệu hành vi khách hàng từ:
      - ratings.json  : {CustomerID, ProductID, Rate, CreateDate}
      - products.json : {Id, Name, UnitPrice}
      - customers.json: {Id, NickName}

    Chiến lược xây dựng chuỗi thời gian:
    ─────────────────────────────────────
    Tập hợp toàn bộ ratings → nhóm theo ngày (hoặc giờ) →
    tạo time-series mức hệ thống:
      * daily_rating_count  : số lượng đánh giá trong ngày
      * daily_avg_rating    : điểm đánh giá trung bình ngày
      * daily_unique_users  : số khách hàng hoạt động
      * daily_unique_products : số sản phẩm được đánh giá
      * daily_high_ratings  : tỷ lệ ratings >= 4
      * daily_low_ratings   : tỷ lệ ratings <= 2
      * rolling_avg_7d      : avg_rating trung bình 7 ngày
      * rolling_count_7d    : rating_count trung bình 7 ngày

    Biến mục tiêu: daily_avg_rating (ngày tiếp theo)

    Parameters
    ----------
    window_size            : độ dài cửa sổ trượt (số ngày/giờ)
    train_ratio            : tỷ lệ train (time-based)
    min_ratings_per_user   : ngưỡng lọc user có đủ dữ liệu (cho per-user stats)
    aggregation            : 'daily' (theo ngày) hoặc 'hourly' (theo giờ)
    top_n_users            : giới hạn số lượng user để tính per-user stats (None = tất cả)

    Returns
    -------
    dict với keys:
      X_train, y_train, X_test, y_test  – numpy arrays
      target_scaler                      – MinMaxScaler cho avg_rating
      feature_names                      – tên các đặc trưng
      df_agg                             – DataFrame chuỗi thời gian đã tổng hợp
      df_ratings                         – DataFrame ratings gốc (đã parse date)
      df_products                        – DataFrame products
      train_size                         – số mẫu train (trước windowing)
      stats                              – thống kê tổng quan
    """
    # ── Đọc file ──
    if not os.path.exists(RATINGS_JSON):
        raise FileNotFoundError(f"Không tìm thấy: {RATINGS_JSON}")

    print("[CustomerData] Đang tải ratings.json ...")
    with open(RATINGS_JSON, "r", encoding="utf-8") as f:
        ratings_raw = json.load(f)

    print("[CustomerData] Đang tải products.json ...")
    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products_raw = json.load(f)

    print("[CustomerData] Đang tải customers.json ...")
    with open(CUSTOMERS_JSON, "r", encoding="utf-8") as f:
        customers_raw = json.load(f)

    # ── Tạo DataFrame ──
    df_ratings  = pd.DataFrame(ratings_raw)
    df_products = pd.DataFrame(products_raw).rename(columns={"Id": "ProductID", "Name": "ProductName"})
    df_customers= pd.DataFrame(customers_raw).rename(columns={"Id": "CustomerID"})

    # Parse datetime
    df_ratings["CreateDate"] = pd.to_datetime(
        df_ratings["CreateDate"], format="%Y/%m/%d %H:%M:%S"
    )
    df_ratings["Rate"] = df_ratings["Rate"].astype(float)

    # Merge thông tin sản phẩm
    df_ratings = df_ratings.merge(
        df_products[["ProductID", "UnitPrice"]], on="ProductID", how="left"
    )
    df_ratings["UnitPrice"] = df_ratings["UnitPrice"].fillna(df_ratings["UnitPrice"].median())

    print(f"[CustomerData] Tổng ratings: {len(df_ratings):,} | "
          f"Khách hàng: {df_ratings['CustomerID'].nunique()} | "
          f"Sản phẩm: {df_ratings['ProductID'].nunique()} | "
          f"Khoảng thời gian: {df_ratings['CreateDate'].min().date()} → "
          f"{df_ratings['CreateDate'].max().date()}")

    # ── Tổng hợp theo ngày (hoặc giờ) ──
    freq_col = "date" if aggregation == "daily" else "datetime_hour"
    if aggregation == "daily":
        df_ratings["period"] = df_ratings["CreateDate"].dt.date
    else:
        df_ratings["period"] = df_ratings["CreateDate"].dt.floor("H")

    df_ratings["period"] = pd.to_datetime(df_ratings["period"])

    agg = df_ratings.groupby("period").agg(
        rating_count    = ("Rate",       "count"),
        avg_rating      = ("Rate",       "mean"),
        unique_users    = ("CustomerID", "nunique"),
        unique_products = ("ProductID",  "nunique"),
        high_ratings    = ("Rate",       lambda x: (x >= 4).sum()),
        low_ratings     = ("Rate",       lambda x: (x <= 2).sum()),
        avg_price       = ("UnitPrice",  "mean"),
        std_rating      = ("Rate",       "std"),
    ).reset_index()

    agg.rename(columns={"period": "date"}, inplace=True)
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)

    # Tỷ lệ
    agg["high_ratio"]    = agg["high_ratings"] / agg["rating_count"].clip(lower=1)
    agg["low_ratio"]     = agg["low_ratings"]  / agg["rating_count"].clip(lower=1)
    agg["std_rating"]    = agg["std_rating"].fillna(0)

    # Rolling features
    agg["rolling_avg_7d"]   = agg["avg_rating"].rolling(7, min_periods=1).mean()
    agg["rolling_count_7d"] = agg["rating_count"].rolling(7, min_periods=1).mean()
    agg["rolling_std_7d"]   = agg["avg_rating"].rolling(7, min_periods=1).std().fillna(0)

    # Lag features
    agg["lag1_avg_rating"]   = agg["avg_rating"].shift(1)
    agg["lag1_rating_count"] = agg["rating_count"].shift(1)
    agg["lag7_avg_rating"]   = agg["avg_rating"].shift(7)

    # Day of week (cyclical encoding)
    dow = agg["date"].dt.dayofweek   # 0=Mon, 6=Sun
    agg["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    agg["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month (cyclical)
    month = agg["date"].dt.month
    agg["month_sin"] = np.sin(2 * np.pi * month / 12)
    agg["month_cos"] = np.cos(2 * np.pi * month / 12)

    agg.dropna(inplace=True)
    agg.reset_index(drop=True, inplace=True)

    # ── Features & target ──
    feature_names = [
        "rating_count", "avg_rating", "unique_users", "unique_products",
        "high_ratio", "low_ratio", "avg_price", "std_rating",
        "rolling_avg_7d", "rolling_count_7d", "rolling_std_7d",
        "lag1_avg_rating", "lag1_rating_count", "lag7_avg_rating",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    target_col  = "avg_rating"
    target_idx  = feature_names.index(target_col)

    data = agg[feature_names].values

    # ── Time-based split ──
    train_size  = int(len(data) * train_ratio)
    train_raw   = data[:train_size]
    test_raw    = data[train_size:]

    # ── Chuẩn hóa (fit TRÊN TRAIN) ──
    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    # ── Sliding Window ──
    X_train, y_train = create_sliding_windows(train_scaled, target_idx, window_size)
    X_test,  y_test  = create_sliding_windows(test_scaled,  target_idx, window_size)

    # ── Target scaler ──
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_raw[:, target_idx : target_idx + 1])

    # ── Thống kê ──
    stats = {
        "total_ratings":    len(df_ratings),
        "total_customers":  df_ratings["CustomerID"].nunique(),
        "total_products":   df_ratings["ProductID"].nunique(),
        "date_range":       f"{df_ratings['CreateDate'].min().date()} → {df_ratings['CreateDate'].max().date()}",
        "avg_rating_global":round(df_ratings["Rate"].mean(), 3),
        "n_periods":        len(agg),
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
        "n_features":       len(feature_names),
    }

    print(f"[CustomerData] Chuỗi thời gian {aggregation}: {len(agg)} mẫu | "
          f"Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Features: {len(feature_names)} | Window: {window_size}")

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_test":        X_test,
        "y_test":        y_test,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "df_agg":        agg,
        "df_ratings":    df_ratings,
        "df_products":   df_products,
        "train_size":    train_size,
        "stats":         stats,
        "aggregation":   aggregation,
    }


# ─────────────────────────────────────────────────────────────
# 5. Lấy danh sách tickers khả dụng (chứng khoán)
# ─────────────────────────────────────────────────────────────
def get_available_tickers(min_rows: int = 500) -> list:
    if not os.path.exists(ALL_INDICES_CSV):
        return []
    df = pd.read_csv(ALL_INDICES_CSV, usecols=["ticker"])
    counts = df["ticker"].value_counts()
    return sorted(counts[counts >= min_rows].index.tolist())


# ─────────────────────────────────────────────────────────────
# 6. Dữ liệu giả lập (fallback / tương thích ngược)
# ─────────────────────────────────────────────────────────────
def generate_stock_market_data(out_path: str = None, n_days: int = 1000,
                               seed: int = 42) -> str:
    if out_path is None:
        out_dir  = os.path.join(_BASE_DIR, "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "stock_prices.csv")

    rng = np.random.default_rng(seed)
    dates  = pd.date_range("2020-01-01", periods=n_days, freq="B")
    close  = 100 + np.cumsum(rng.normal(0.05, 1.0, n_days))
    df = pd.DataFrame({
        "date":   dates,
        "open":   close + rng.normal(0, 1.0, n_days),
        "high":   close + rng.uniform(0.5, 3.0, n_days),
        "low":    close - rng.uniform(0.5, 3.0, n_days),
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
    })
    df = _add_technical_indicators(df)
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Stock giả lập: {out_path} ({len(df)} hàng)")
    return out_path


def generate_customer_behavior_data(out_path: str = None, **kwargs) -> str:
    """Giữ lại để tương thích ngược. Data thực tế dùng load_customer_behavior_data()."""
    if out_path is None:
        out_dir  = os.path.join(_BASE_DIR, "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "customer_behavior.csv")
    print(f"[DataGen] Sử dụng load_customer_behavior_data() cho data thực tế.")
    return out_path


# ─────────────────────────────────────────────────────────────
# 7. Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TEST 1: Dữ liệu Chứng Khoán")
    print("=" * 55)
    tickers = get_available_tickers()
    print(f"Tickers khả dụng ({len(tickers)}): {tickers[:8]} ...")
    result_stock = load_real_stock_data(ticker="^GSPC", window_size=20)
    print(f"  X_train: {result_stock['X_train'].shape}")

    print("\n" + "=" * 55)
    print("  TEST 2: Dữ liệu Hành vi Khách hàng")
    print("=" * 55)
    result_cust = load_customer_behavior_data(window_size=10, aggregation="daily")
    print(f"  X_train: {result_cust['X_train'].shape}")
    print(f"  Features: {result_cust['feature_names']}")
    print(f"  Stats: {result_cust['stats']}")
