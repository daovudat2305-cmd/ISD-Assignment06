# ISD-Assignment06: Recurrent Neural Networks (Vanilla RNN) cho Dự Báo Chuỗi Thời Gian

Dự án nghiên cứu, hiện thực và đánh giá chuyên sâu mô hình **Vanilla RNN (Mạng Nơ-ron Hồi quy cơ bản)** trên 2 tập dữ liệu thực tế:
1. **Dự báo giá chứng khoán** (36 chỉ số toàn cầu, tập trung S&P 500 `^GSPC`).
2. **Dự báo hành vi khách hàng** (100k+ ratings từ hệ thống bán lẻ giai đoạn 2018–2022).

Dự án được xây dựng song song trên cả hai framework Deep Learning hàng đầu: **PyTorch** (`nn.RNN`) và **TensorFlow / Keras** (`layers.SimpleRNN`).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
ISD-Assignment06/
├── README.md
├── requirements.txt
├── data/
│   ├── stock_price/
│   │   ├── all_indices_data.csv              ← Dữ liệu chứng khoán THỰC TẾ (36 chỉ số)
│   │   └── individual_indices_data/          ← File riêng từng chỉ số (^GSPC, ^DJI,...)
│   └── customer_behavior/
│       ├── ratings.json                      ← Lịch sử đánh giá 100k+ bản ghi
│       ├── products.json                     ← Danh mục 691 sản phẩm
│       └── customers.json                    ← Thông tin khách hàng
├── src/
│   ├── __init__.py
│   ├── data_generator.py      ← Load & tiền xử lý dữ liệu thực, sinh đặc trưng chuỗi
│   ├── rnn_scratch.py         ← Vanilla RNN + BPTT thuần Python/NumPy (không thư viện)
│   ├── stock_pytorch.py       ← Pipeline PyTorch Stock: Vanilla RNN, Train, Eval, Viz
│   ├── stock_keras.py         ← Pipeline Keras Stock: SimpleRNN, Callbacks, Viz
│   ├── customer_pytorch.py    ← Pipeline PyTorch Customer: Vanilla RNN, Train, Eval, Viz
│   └── customer_keras.py      ← Pipeline Keras Customer: SimpleRNN, Callbacks, Viz
├── notebooks/
│   ├── 01_stock_price_rnn.ipynb         ← Dự báo giá chứng khoán bằng Vanilla RNN
│   └── 02_customer_behavior_rnn.ipynb   ← Dự báo hành vi khách hàng bằng Vanilla RNN
└── reports/
    ├── assignment06_report_part1.md     ← Báo cáo lý thuyết & phân tích dữ liệu chi tiết
    └── figures/                         ← Biểu đồ EDA, Loss curves, Predictions, Framework Comparison
```

---

## 📊 Hai Tập Dữ Liệu Thực Tế

### Dataset 1: Chứng Khoán Toàn Cầu (Stock Price Time-Series)
- **Nguồn:** 36 chỉ số tài chính quốc tế (S&P 500, Dow Jones, NASDAQ, Nikkei 225, FTSE 100,...).
- **Mục tiêu:** Dự đoán giá đóng cửa (`close`) phiên tiếp theo dựa trên cửa sổ trượt $T=20$ phiên.
- **11 Đặc trưng kỹ thuật:** `open`, `high`, `low`, `close`, `volume`, `return`, `ma7`, `ma21`, `volatility_7d`, `rsi_14`, `bb_width`.

### Dataset 2: Hành Vi Đánh Giá Khách Hàng (Customer Rating Behavior)
- **Nguồn:** Hệ thống nhật ký giao dịch thương mại điện tử (~700 khách hàng, 691 sản phẩm).
- **Mục tiêu:** Dự đoán điểm đánh giá trung bình (`avg_rating`) của toàn hệ thống ngày tiếp theo ($T=10$ ngày).
- **18 Đặc trưng kỹ thuật:**
  - Thống kê ngày: `rating_count`, `avg_rating`, `unique_users`, `unique_products`, `high_ratio`, `low_ratio`, `avg_price`, `std_rating`.
  - Xu hướng & Trễ: `rolling_avg_7d`, `rolling_count_7d`, `rolling_std_7d`, `lag1_avg_rating`, `lag1_rating_count`, `lag7_avg_rating`.
  - Tuần hoàn (Cyclical): `dow_sin`, `dow_cos`, `month_sin`, `month_cos`.

---

## 🧠 Cơ Sở Lý Thuyết: Vanilla Recurrent Neural Network (RNN)

### 1. Bản Chất Trạng Thái Ẩn (Hidden State)
Các mạng truyền thẳng (MLP/CNN) giả định dữ liệu I.I.D. (độc lập), không có khả năng ghi nhớ trật tự chuỗi thời gian. Vanilla RNN giải quyết bằng cơ chế **Hidden State $h_t$** tích lũy thông tin quá khứ:

$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
$$h_t = \tanh(a_t)$$
$$\hat{y}_t = W_{hy} h_t + b_y$$

Trong đó ma trận hồi quy $W_{hh}$ được chia sẻ qua tất cả các bước thời gian (Parameter Sharing Across Time).

### 2. Thuật Toán BPTT & Thách Thức Vanishing Gradient
Khi lan truyền ngược qua thời gian (Backpropagation Through Time):

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^T \left[\text{diag}(1-h_t^2) \cdot W_{hh}^T\right]$$

Khi $\|W_{hh}\| < 1$ và đạo hàm $\tanh \in (0, 1]$, tích liên tiếp qua nhiều bước thời gian khiến gradient tiệm cận về 0 (**Vanishing Gradient**), làm RNN suy giảm khả năng liên kết thông tin dài hạn.

### 3. Giải Pháp Tối Ưu Cho Vanilla RNN Trong Dự Án
1. **Thiết lập Sliding Window ngắn ($T = 10 \sim 20$):** Giảm số bước đạo hàm ngược, bảo toàn dòng gradient.
2. **Kỹ nghệ đặc trưng giàu thông tin (Feature Engineering):** Bổ sung các biến Lag (`lag1`, `lag7`) và Rolling averages (`ma7`, `ma21`) để đưa thông tin chu kỳ dài hạn trực tiếp vào vector $x_t$ ở từng bước.
3. **Gradient Clipping (`max_norm=1.0`):** Ngăn chặn hiện tượng bùng nổ gradient (Exploding Gradient).
4. **Batch Normalization (Keras):** Chuẩn hóa activations giữa các lớp SimpleRNN, ổn định quá trình học.
5. **Learning Rate Scheduler & Early Stopping:** `ReduceLROnPlateau` tự động giảm learning rate khi loss hội tụ.

---

## 💻 Kiến Trúc Mô Hình Trong Dự Án

### PyTorch Pipeline (`src/stock_pytorch.py`, `src/customer_pytorch.py`)
```text
[Input (B, T, Features)] 
  → [nn.RNN(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)]
  → [Linear(hidden_dim → 64) → ReLU → Dropout]
  → [Linear(64 → 32) → ReLU]
  → [Linear(32 → 1)]   # Dự đoán giá trị liên tục
```

### Keras Pipeline (`src/stock_keras.py`, `src/customer_keras.py`)
```text
Sequential([
    SimpleRNN(units, return_sequences=True, input_shape=(T, Features)),
    Dropout(0.25), BatchNormalization(),
    SimpleRNN(units // 2, return_sequences=False),
    Dropout(0.25), BatchNormalization(),
    Dense(64, activation="relu"), Dropout(0.125),
    Dense(32, activation="relu"),
    Dense(1, name="output")
])
```

---

## 🚀 Hướng Dẫn Chạy & Thực Nghiệm

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy demo Vanilla RNN thuần NumPy (Hiểu sâu bản chất toán học)
```powershell
python src/rnn_scratch.py
```

### Bước 3: Chạy Pipeline PyTorch (Dự báo Stock & Customer)
```powershell
python src/stock_pytorch.py
python src/customer_pytorch.py
```

### Bước 4: Chạy Pipeline Keras (Dự báo Stock & Customer)
```powershell
python src/stock_keras.py
python src/customer_keras.py
```

### Bước 5: Mở 2 Jupyter Notebooks để theo dõi trực quan
```powershell
jupyter lab notebooks/01_stock_price_rnn.ipynb
jupyter lab notebooks/02_customer_behavior_rnn.ipynb
```

---

## 📈 Chỉ Số Đánh Giá (Metrics)

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

---

## 📂 Biểu Đồ Đầu Ra (`reports/figures/`)

| File Biểu Đồ | Nội Dung Chi Tiết |
|--------------|-------------------|
| `eda_{ticker}_pytorch.png` | Phân tích EDA dữ liệu chứng khoán (6 panels: Giá, MA, RSI, Bollinger, Heatmap) |
| `stock_{ticker}_rnn_pytorch.png` | Đồ thị huấn luyện Vanilla RNN PyTorch (Loss, Actual vs Pred, Residuals) |
| `stock_{ticker}_rnn_keras.png` | Đồ thị huấn luyện SimpleRNN Keras (MSE, MAE, Actual vs Pred, Residuals) |
| `stock_framework_{ticker}.png` | So sánh trực diện hiệu năng PyTorch RNN vs Keras RNN (Stock) |
| `customer_eda_pytorch.png` | Phân tích EDA hành vi khách hàng 8-panel đầy đủ |
| `customer_rnn_pytorch.png` | Đồ thị huấn luyện Vanilla RNN PyTorch trên dữ liệu khách hàng |
| `customer_rnn_keras.png` | Đồ thị huấn luyện SimpleRNN Keras trên dữ liệu khách hàng |
| `customer_framework.png` | So sánh trực diện hiệu năng PyTorch RNN vs Keras RNN (Customer Behavior) |
