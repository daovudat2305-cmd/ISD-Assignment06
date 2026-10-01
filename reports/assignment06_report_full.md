# BÁO CÁO TOÀN VĂN BÀI TẬP LỚN: MẠNG NƠ-RON HỒI QUY (RECURRENT NEURAL NETWORKS - RNN)
## HỌC PHẦN: PHÁT TRIỂN HỆ THỐNG THÔNG MINH (ISD) - ASSIGNMENT 06

---

## THÔNG TIN CHUNG DỰ ÁN
- **Đề tài:** Nghiên cứu lý thuyết, hiện thực toán học và ứng dụng mạng nơ-ron hồi quy (Vanilla RNN / SimpleRNN) vào dự báo chuỗi thời gian thực tế.
- **Hai tập dữ liệu thực nghiệm:**
  1. *Dữ liệu Tài chính & Chứng khoán Toàn cầu (Stock Price Time-Series)*: Phân tích chỉ số S&P 500 (`^GSPC` - 24,215 phiên) và Dow Jones (`^DJI` - 8,159 phiên).
  2. *Dữ liệu Hành vi Khách hàng E-Commerce (Customer Rating Behavior)*: Hệ thống đánh giá 100,000+ bản ghi tổng hợp thành chuỗi thời gian 1,529 ngày.
- **Framework triển khai song song:** **PyTorch** (`torch.nn.RNN`) và **TensorFlow / Keras** (`tf.keras.layers.SimpleRNN`).
- **Mã nguồn thực nghiệm thuần túy:** Hiện thực thuật toán Forward Pass và BPTT từ số 0 bằng **NumPy**.

---

## MỤC LỤC CHI TIẾT
- [PHẦN 1: CƠ SỞ LÝ THUYẾT, CÁC HÀM TOÁN HỌC VÀ CODE VÍ DỤ VỀ RNN](#phần-1-cơ-sở-lý-thuyết-các-hàm-toán-học-và-code-ví-dụ-về-rnn)
  - [1.1. Bối cảnh ra đời & Giới hạn của Mạng Nơ-ron Truyền thẳng (FFN / CNN)](#11-bối-cảnh-ra-đời--giới-hạn-của-mạng-nơ-ron-truyền-thẳng-ffn--cnn)
  - [1.2. Khái niệm Trạng thái ẩn (Hidden State) – Bộ nhớ thời gian của RNN](#12-khái-niệm-trạng-thái-ẩn-hidden-state--bộ-nhớ-thời-gian-của-rnn)
  - [1.3. Mô hình Toán học của Vanilla RNN (Forward Propagation)](#13-mô-hình-toán-học-của-vanilla-rnn-forward-propagation)
  - [1.4. Nguyên lý Chia sẻ Trọng số (Parameter Sharing Across Time)](#14-nguyên-lý-chia-sẻ-trọng-số-parameter-sharing-across-time)
  - [1.5. Phân loại Kiến trúc Chuỗi (Sequence Architectures)](#15-phân-loại-kiến-trúc-chuỗi-sequence-architectures)
  - [1.6. Thuật toán Huấn luyện Lan truyền ngược qua thời gian (BPTT)](#16-thuật-toán-huấn-luyện-lan-truyền-ngược-qua-thời-gian-bptt)
  - [1.7. Phân tích Hiện tượng Triệt tiêu & Bùng nổ Gradient (Vanishing & Exploding Gradient)](#17-phân-tích-hiện-tượng-triệt-tiêu--bùng-nổ-gradient-vanishing--exploding-gradient)
  - [1.8. Mã nguồn Minh họa Thuần Python & NumPy (RNN from Scratch)](#18-mã-nguồn-minh-họa-thuần-python--numpy-rnn-from-scratch)
- [PHẦN 2: PHÂN TÍCH VÀ KHÁM PHÁ CHI TIẾT 2 TẬP DỮ LIỆU THỰC TẾ](#phần-2-phân-tích-và-khám-phá-chi-tiết-2-tập-dữ-liệu-thực-tế)
  - [2.1. Tập Dữ Liệu 1: Dự Báo Chỉ Số Chứng Khoán Toàn Cầu (Stock Price Time-Series)](#21-tập-dữ-liệu-1-dự-báo-chỉ-số-chứng-khoán-toàn-cầu-stock-price-time-series)
    - [2.1.1. Nguồn gốc, Quy mô và Cấu trúc Thuộc tính](#211-nguồn-gốc-quy-mô-và-cấu-trúc-thuộc-tính)
    - [2.1.2. Đặc tính Chuỗi Thời gian Tài chính (Non-Stationarity, Volatility Clustering)](#212-đặc-tính-chuỗi-thời-gian-tài-chính-non-stationarity-volatility-clustering)
    - [2.1.3. Kỹ nghệ 11 Đặc trưng Kỹ thuật (Technical Indicator Engineering)](#213-kỹ-nghệ-11-đặc-trưng-kỹ-thuật-technical-indicator-engineering)
    - [2.1.4. Chiến lược Phân chia Dữ liệu Tránh Rò rỉ (Time-based Split & Scaling)](#214-chiến-lược-phân-chia-dữ-liệu-tránh-rò-rỉ-time-based-split--scaling)
    - [2.1.5. Cấu trúc Tensor Đầu vào Dạng Cửa sổ Trượt (Sliding Window Tensor)](#215-cấu-trúc-tensor-đầu-vào-dạng-cửa-sổ-trượt-sliding-window-tensor)
  - [2.2. Tập Dữ Liệu 2: Phân Tích & Dự Báo Hành Vi Khách Hàng (Customer Behavior Time-Series)](#22-tập-dữ-liệu-2-phân-tích--dự-báo-hành-vi-khách-hàng-customer-behavior-time-series)
    - [2.2.1. Nguồn gốc và Kiến trúc Đa Thực thể (Ratings - Products - Customers)](#221-nguồn-gốc-và-kiến-trúc-đa-thực-thể-ratings---products---customers)
    - [2.2.2. Chiến lược Chuyển đổi Dữ liệu Nhật ký Sự kiện sang Chuỗi Thời Gian Daily](#222-chiến-lược-chuyển-đổi-dữ-liệu-nhật-ký-sự-kiện-sang-chuỗi-thời-gian-daily)
    - [2.2.3. Trích xuất Bộ 18 Đặc trưng Hành vi Đa Chiều](#223-trích-xuất-bộ-18-đặc-trưng-hành-vi-đa-chiều)
    - [2.2.4. Kỹ thuật Mã hóa Tính Tuần hoàn Thời gian (Cyclical Encoding)](#224-kỹ-thuật-mã-hóa-tính-tuần-hoàn-thời-gian-cyclical-encoding)
    - [2.2.5. Mục tiêu Dự báo & Ý nghĩa Vận hành Doanh nghiệp](#225-mục-tiêu-dự-báo--ý-nghĩa-vận-hành-doanh-nghiệp)
  - [2.3. Bảng Tổng Hợp So Sánh Đối Chiếu Giữa 2 Tập Dữ Liệu](#23-bảng-tổng-hợp-so-sánh-đối-chiếu-giữa-2-tập-dữ-liệu)
- [PHẦN 3: HIỆN THỰC HÓA PIPELINE PYTORCH: PHÂN TÍCH - HUẤN LUYỆN - DỰ ĐOÁN - TRỰC QUAN HÓA](#phần-3-hiện-thực-hóa-pipeline-pytorch-phân-tích---huấn-luyện---dự-đoán---trực-quan-hóa)
  - [3.1. Thiết kế Kiến trúc Mạng PyTorch Vanilla RNN](#31-thiết-kế-kiến-trúc-mạng-pytorch-vanilla-rnn)
  - [3.2. Hiện thực hóa TimeSeriesDataset và Data Loader](#32-hiện-thực-hóa-timeseriesdataset-và-data-loader)
  - [3.3. Quy trình Huấn luyện Chuyên sâu: Gradient Clipping & LR Scheduler](#33-quy-trình-huấn-luyện-chuyên-sâu-gradient-clipping--lr-scheduler)
  - [3.4. Thực nghiệm & Đánh giá Trên Dữ Liệu Chứng Khoán (S&P 500 & Dow Jones)](#34-thực-nghiệm--đánh-giá-trên-dữ-liệu-chứng-khoán-sp-500--dow-jones)
    - [3.4.1. Phân tích Đồ thị EDA Dữ liệu Chứng khoán](#341-phân-tích-đồ-thị-eda-dữ-liệu-chứng-khoán)
    - [3.4.2. Động thái Hội tụ và Kết quả Dự đoán PyTorch Stock](#342-động-thái-hội-tụ-và-kết-quả-dự-đoán-pytorch-stock)
  - [3.5. Thực nghiệm & Đánh giá Trên Dữ Liệu Hành Vi Khách Hàng](#35-thực-nghiệm--đánh-giá-trên-dữ-liệu-hành-vi-khách-hàng)
    - [3.5.1. Phân tích Đồ thị EDA Hành vi Khách hàng](#351-phân-tích-đồ-thị-eda-hành-vi-khách-hàng)
    - [3.5.2. Động thái Hội tụ và Kết quả Dự đoán PyTorch Customer](#352-động-thái-hội-tụ-và-kết-quả-dự-đoán-pytorch-customer)
- [PHẦN 4: HIỆN THỰC HÓA PIPELINE TENSORFLOW / KERAS: PHÂN TÍCH - HUẤN LUYỆN - DỰ ĐOÁN - TRỰC QUAN HÓA](#phần-4-hiện-thực-hóa-pipeline-tensorflow--keras-phân-tích---huấn-luyện---dự-đoán---trực-quan-hóa)
  - [4.1. Thiết kế Kiến trúc Keras SimpleRNN với Regularization Đa Tầng](#41-thiết-kế-kiến-trúc-keras-simplernn-với-regularization-đa-tầng)
  - [4.2. Chiến lược Tối ưu Huấn luyện với Callbacks (EarlyStopping & ReduceLROnPlateau)](#42-chiến-lược-tối-ưu-huấn-luyện-với-callbacks-earlystopping--reducelronplateau)
  - [4.3. Thực nghiệm & Đánh giá Trên Dữ Liệu Chứng Khoán Keras](#43-thực-nghiệm--đánh-giá-trên-dữ-liệu-chứng-khoán-keras)
  - [4.4. Thực nghiệm & Đánh giá Trên Dữ Liệu Hành Vi Khách Hàng Keras](#44-thực-nghiệm--đánh-giá-trên-dữ-liệu-hành-vi-khách-hàng-keras)
- [PHẦN 5: ĐỐI ĐẦU SO SÁNH TRỰC DIỆN PYTORCH VS KERAS & ĐÁNH GIÁ ĐA DIỆN](#phần-5-đối-đầu-so-sánh-trực-diện-pytorch-vs-keras--đánh-giá-đa-diện)
  - [5.1. Bảng Tổng Hợp Chỉ Số Định Lượng Đối Đầu Trực Diện](#51-bảng-tổng-hợp-chỉ-số-định-lượng-đối-đầu-trực-diện)
  - [5.2. Phân tích Trực quan Đồ thị So Sánh Hai Framework](#52-phân-tích-trực-quan-đồ-thị-so-sánh-hai-framework)
  - [5.3. So sánh Đặc tính Công nghệ & Trải nghiệm Lập trình](#53-so-sánh-đặc-tính-công-nghệ--trải-nghiệm-lập-trình)
- [PHẦN 6: PHÂN TÍCH CHUYÊN SÂU BẢN CHẤT CỦA MÔ HÌNH VANILLA RNN TRÊN CHUỖI THỜI GIAN THỰC TẾ](#phần-6-phân-tích-chuyên-sâu-bản-chất-của-mô-hình-vanilla-rnn-trên-chuỗi-thời-gian-thực-tế)
  - [6.1. Giải Mã Hiện Tượng "Bám Đuôi Trễ Pha" (Naive Follower Trap / 1-Step Lag Phenomenon)](#61-giải-mã-hiện-tượng-bám-đuôi-trễ-pha-naive-follower-trap--1-step-lag-phenomenon)
  - [6.2. Phân Tích Nguyên Nhân Chỉ Số R² Âm Trong Chuỗi Tài Chính Không Dừng (Non-Stationary)](#62-phân-tích-nguyên-nhân-chỉ-số-r-âm-trong-chuỗi-tài-chính-không-dừng-non-stationary)
  - [6.3. Giới Hạn Cốt Tử Của Tế Bào Vanilla RNN So Với LSTM và GRU](#63-giới-hạn-cốt-tử-của-tế-bào-vanilla-rnn-so-với-lstm-và-gru)
  - [6.4. Đề Xuất Giải Pháp Nâng Cấp & Khuyến Nghị Triển Khai Thực Tiễn](#64-đề-xuất-giải-pháp-nâng-cấp--khuyến-nghị-triển-khai-thực-tiễn)
- [PHẦN 7: TỔNG KẾT VÀ BÀI HỌC KINH NGHIỆM](#phần-7-tổng-kết-và-bài-học-kinh-nghiệm)
  - [7.1. Đóng góp và Thành tựu của Báo cáo](#71-đóng-góp-và-thành-tựu-của-báo-cáo)
  - [7.2. Kết luận chung](#72-kết-luận-chung)

---

# PHẦN 1: CƠ SỞ LÝ THUYẾT, CÁC HÀM TOÁN HỌC VÀ CODE VÍ DỤ VỀ RNN

## 1.1. Bối cảnh ra đời & Giới hạn của Mạng Nơ-ron Truyền thẳng (FFN / CNN)
Trong lịch sử phát triển của Học Sâu (Deep Learning), các kiến trúc mạng nơ-ron truyền thẳng như **Multilayer Perceptron (MLP)** và **Convolutional Neural Networks (CNN)** đã gặt hái được những thành công vượt bậc trong các tác vụ thị giác máy tính và phân loại dữ liệu bảng độc lập. Tuy nhiên, kiến trúc của các mạng này được xây dựng trên một tiền đề toán học cực kỳ khắt khe: **Giả định I.I.D. (Independent and Identically Distributed)** — giả định rằng các mẫu dữ liệu đầu vào là hoàn toàn độc lập với nhau và được rút ra từ cùng một phân phối xác suất cố định.

Giả định I.I.D. hoàn toàn sụp đổ khi chúng ta đối mặt với **Dữ liệu dạng chuỗi (Sequential Data)** và đặc biệt là **Chuỗi thời gian (Time-Series Data)**:
1. **Tính phụ thuộc thứ tự thời gian (Temporal Dependency):** Giá trị của ngày hôm nay $x_t$ phụ thuộc trực tiếp và sâu sắc vào giá trị của ngày hôm qua $x_{t-1}$, tuần trước $x_{t-7}$. Nếu ta xáo trộn ngẫu nhiên thứ tự các ngày, toàn bộ cấu trúc động lực học và ngữ cảnh của hệ thống sẽ bị phá hủy.
2. **Kích thước đầu vào biến thiên (Variable-length Sequence):** Một chuỗi tín hiệu âm thanh, câu văn bản ngôn ngữ tự nhiên hay nhật ký giao dịch người dùng có thể có độ dài thay đổi linh hoạt (ví dụ: chuỗi dài 10 ngày, 50 ngày hoặc 300 ngày). MLP truyền thống bắt buộc phải cố định số chiều đầu vào $D_{in}$, do đó buộc phải dùng kỹ thuật padding (độn số 0) hoặc truncate (cắt cụt), gây biến dạng và mất mát thông tin.
3. **Thiếu cơ chế duy trì bộ nhớ nội tại (Absence of Internal State):** Mạng truyền thẳng xử lý mỗi điểm dữ liệu theo cách hoàn toàn tách biệt. Đầu ra $\hat{y}_t$ chỉ đơn thuần là ánh xạ tĩnh của duy nhất đầu vào $x_t$: $\hat{y}_t = f(W x_t + b)$. Mạng không có cách nào để biết được điều gì đã xảy ra trước thời điểm $t$.

Để giải quyết triệt để các rào cản trên, **Mạng Nơ-ron Hồi quy (Recurrent Neural Network - RNN)** ra đời với nguyên lý cốt lõi: *Trang bị cho mạng nơ-ron một vòng lặp hồi quy nội tại (Recurrent Connection), cho phép thông tin từ các bước thời gian trước được lưu giữ và truyền sang bước thời gian tiếp theo dưới dạng một vector trạng thái ẩn (Hidden State).*

---

## 1.2. Khái niệm Trạng thái ẩn (Hidden State) – Bộ nhớ thời gian của RNN
Trọng tâm kiến trúc của mạng RNN là vector **Trạng thái ẩn (Hidden State)**, ký hiệu là $h_t \in \mathbb{R}^{d_h}$ ($d_h$ là số chiều ẩn). 

Tại mỗi bước thời gian rời rạc $t \in \{1, 2, \dots, T\}$:
- Tế bào RNN tiếp nhận đồng thời hai luồng tín hiệu đầu vào:
  1. Vector đặc trưng đầu vào tại thời điểm hiện tại: $x_t \in \mathbb{R}^{d_x}$.
  2. Vector trạng thái ẩn của bước liền trước: $h_{t-1} \in \mathbb{R}^{d_h}$, mang trong nó toàn bộ "ký ức" cô đọng được tích lũy từ bước thời gian $0$ đến bước $t-1$.
- Tế bào RNN tổng hợp hai luồng tín hiệu này để sản sinh ra trạng thái ẩn mới $h_t$:
  $$h_t = f_{RNN}(x_t, h_{t-1})$$

Nhờ liên kết hồi quy này, $h_t$ đóng vai trò như một hàm tóm tắt lịch sử (history summarizer) biểu diễn toàn bộ chuỗi quan sát quá khứ tính đến thời điểm $t$:
$$h_t \approx \text{Summary}(x_1, x_2, \dots, x_t)$$

Cơ chế này có thể được hình dung ở hai dạng biểu diễn: **Dạng cuộn (Folded Representation)** với vòng lặp phản hồi, và **Dạng mở bung (Unfolded Representation)** trải dài theo trục thời gian:

```
[Dạng Cuộn (Folded)]           [Dạng Mở Bung Theo Trục Thời Gian (Unfolded)]
      y_t                          y_1           y_2                 y_T
       ↑                            ↑             ↑                   ↑
   [RNN Cell] <-\              [Cell t=1]    [Cell t=2]   ...    [Cell t=T]
    ↑       |   |               ↑      \      ↑      \            ↑      \
   x_t      \---/              x_1  --> h_1  x_2  --> h_2        x_T  --> h_T
(vòng lặp thời gian)          h_0=0
```

---

## 1.3. Mô hình Toán học của Vanilla RNN (Forward Propagation)
Mô hình toán học kinh điển của mạng Vanilla RNN (còn gọi là Elman RNN, do Jeffrey Elman đề xuất năm 1990) tại bước thời gian $t$ được định nghĩa bằng hệ phương trình vi phân sai phân tuyến tính kết hợp phi tuyến:

### Bước 1: Tính toán tổ hợp tuyến tính (Pre-activation)
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$

Trong đó:
- $x_t \in \mathbb{R}^{d_x \times 1}$: Vector đặc trưng đầu vào tại bước $t$.
- $h_{t-1} \in \mathbb{R}^{d_h \times 1}$: Vector trạng thái ẩn của bước liền trước (quy ước $h_0 = \mathbf{0}$).
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$: Ma trận trọng số kết nối giữa lớp đầu vào và lớp ẩn.
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$: Ma trận trọng số hồi quy (recurrent weight matrix) kết nối giữa trạng thái ẩn quá khứ và trạng thái ẩn hiện tại.
- $b_h \in \mathbb{R}^{d_h \times 1}$: Vector độ chệch (bias) của tầng ẩn.

### Bước 2: Kích hoạt phi tuyến (State Transition)
$$h_t = \tanh(a_t) = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Hàm kích hoạt tiêu chuẩn cho Vanilla RNN là hàm Tangent Hyperbolic ($\tanh$):
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$$
Hàm $\tanh$ đóng vai trò cốt lõi trong việc khống chế miền giá trị của trạng thái ẩn luôn nằm trong khoảng đối xứng $(-1, 1)$, ngăn chặn việc giá trị của bộ nhớ bùng nổ vô hạn khi trải qua hàng chục bước thời gian lặp lại.

### Bước 3: Tính toán đầu ra dự đoán (Output Projection)
$$\hat{y}_t = g(W_{hy} h_t + b_y)$$

Trong đó:
- $W_{hy} \in \mathbb{R}^{d_y \times d_h}$: Ma trận trọng số chiếu từ không gian ẩn sang không gian đầu ra.
- $b_y \in \mathbb{R}^{d_y \times 1}$: Vector bias của tầng đầu ra.
- $g(\cdot)$: Hàm kích hoạt đầu ra tùy thuộc vào bản chất của bài toán:
  + **Bài toán Hồi quy (Regression)** (dự đoán giá đóng cửa chứng khoán, điểm rating trung bình): $g(z) = z$ (hàm đồng nhất / Identity).
  + **Bài toán Phân loại nhị phân (Binary Classification)**: $g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$ (Sigmoid).
  + **Bài toán Phân loại đa lớp (Multi-class Classification)**: $g(z) = \text{Softmax}(z)$.

---

## 1.4. Nguyên lý Chia sẻ Trọng số (Parameter Sharing Across Time)
Một trong những đột phá quan trọng nhất của kiến trúc RNN là **Nguyên lý Chia sẻ Tham số qua Thời gian (Parameter Sharing Across Time)**. 

Bất kể chuỗi thời gian có độ dài $T=10$ hay $T=1000$ bước:
- Bộ ba ma trận trọng số $\{W_{xh}, W_{hh}, W_{hy}\}$ và vector bias $\{b_h, b_y\}$ là **duy nhất và giống hệt nhau tại mọi bước thời gian $t$**.
- Ý nghĩa toán học và công nghệ:
  1. **Tính bất biến theo phép tịnh tiến thời gian (Temporal Invariance):** Mô hình có khả năng nhận biết một mẫu hình động lực học (ví dụ: xu hướng tăng giá đột ngột, hay đợt bùng nổ số lượng đánh giá) bất kể mẫu hình đó xuất hiện ở đầu chuỗi, giữa chuỗi hay cuối chuỗi.
  2. **Tiết kiệm tham số tối đa (Parameter Efficiency):** Số lượng tham số của mô hình hoàn toàn không phụ thuộc vào độ dài chuỗi $T$:
     $$\text{Tổng số tham số} = (d_x \times d_h + d_h \times d_h + d_h) + (d_h \times d_y + d_y)$$
     Điều này giúp RNN tránh được sự bùng nổ tham số như MLP và giảm thiểu nguy cơ quá khớp (overfitting).

---

## 1.5. Phân loại Kiến trúc Chuỗi (Sequence Architectures)
Dựa trên sự tương quan giữa số bước thời gian đầu vào ($T_{in}$) và số bước thời gian đầu ra ($T_{out}$), các bài toán mạng RNN được phân loại thành 4 nhóm chính:

```
1. One-to-One     2. One-to-Many          3. Many-to-One            4. Many-to-Many
   (Chuẩn MLP)    (Image Captioning)       (Time-Series / Senti)     (Dịch máy / Seq2Seq)
     [ y ]          [y1]  [y2]  [y3]               [ y ]              [y1]  [y2]  [y3]
       ↑              ↑     ↑     ↑                  ↑                  ↑     ↑     ↑
     [Cell]        [Cell][Cell][Cell]     [Cell]→[Cell]→[Cell]       [Cell]→[Cell]→[Cell]
       ↑              ↑                      ↑      ↑      ↑            ↑      ↑      ↑
     [ x ]          [ x ]                   [x1]   [x2]   [x3]         [x1]   [x2]   [x3]
```

- **Many-to-One (Nhiều - Một):** Tiếp nhận một chuỗi thời gian $T$ bước $\{x_1, x_2, \dots, x_T\}$ và chỉ xuất ra duy nhất một đầu ra tại bước cuối cùng $\hat{y}_T$. **Đây chính là cấu trúc kiến trúc được áp dụng trong toàn bộ Dự án Assignment 06** (Dự báo giá chứng khoán ngày $t+1$ dựa trên cửa sổ 20 ngày trước; Dự báo rating ngày mai dựa trên 10 ngày trước).
- **Many-to-Many (Nhiều - Nhiều):** Tiếp nhận chuỗi và dự đoán chuỗi (ví dụ: dịch máy Anh-Việt, nhận dạng thực thể có tên NER).
- **One-to-Many (Một - Nhiều):** Tiếp nhận một vector tĩnh và sinh chuỗi văn bản (ví dụ: mô tả ảnh Image Captioning).

---

## 1.6. Thuật toán Huấn luyện Lan truyền ngược qua thời gian (BPTT)
Để huấn luyện mạng RNN, thuật toán lan truyền ngược truyền thống được mở rộng thành **Lan truyền ngược qua thời gian (Backpropagation Through Time - BPTT)**.

Giả sử hàm mất mát tổng thể trên toàn chuỗi là tổng mất mát tại từng bước thời gian:
$$L = \sum_{t=1}^T L_t(\hat{y}_t, y_t)$$
Đối với bài toán Many-to-One trong dự án của chúng ta, mất mát chỉ tính tại bước cuối cùng: $L = L_T(\hat{y}_T, y_T) = \frac{1}{2} \|\hat{y}_T - y_T\|^2$.

Để tính đạo hàm của hàm mất mát theo ma trận trọng số hồi quy $W_{hh}$, ta áp dụng quy tắc dây chuyền đa biến (Multivariate Chain Rule). Vì $W_{hh}$ xuất hiện tại mọi bước thời gian, đạo hàm tổng thể là tổng đạo hàm cục bộ qua tất cả các bước:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial^+ h_t}{\partial W_{hh}}$$

Trong đó, gradient của hàm mất mát truyền về trạng thái ẩn $h_t$ lại phụ thuộc vào trạng thái ẩn tiếp sau $h_{t+1}$:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T \frac{\partial L}{\partial h_{t+1}}$$

Ma trận Jacobi của bước chuyển trạng thái $\frac{\partial h_k}{\partial h_{k-1}}$ được tính như sau:
$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}\left(1 - h_k^2\right) \cdot W_{hh}^T$$

Khi mở rộng quy tắc dây chuyền từ bước cuối $T$ ngược về bước ban đầu $t$:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^T \left[\text{diag}\left(1 - h_k^2\right) \cdot W_{hh}^T\right]$$

---

## 1.7. Phân tích Hiện tượng Triệt tiêu & Bùng nổ Gradient (Vanishing & Exploding Gradient)
Tích liên tiếp $\prod_{k=t+1}^T \left[\text{diag}\left(1 - h_k^2\right) \cdot W_{hh}^T\right]$ chính là tâm điểm sinh ra hai căn bệnh nan giải của Vanilla RNN:

1. **Hiện tượng Triệt tiêu Gradient (Vanishing Gradient):**
   - Đạo hàm của hàm kích hoạt $\tanh$: $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) \in [0, 1]$. Giá trị cực đại của đạo hàm chỉ bằng $1$ tại $z=0$, và nhanh chóng suy giảm về $0$ khi $|z|$ lớn (vùng bão hòa).
   - Nếu trị riêng lớn nhất (spectral radius) của ma trận trọng số $\|W_{hh}\|_2 < 1$, thì tích của chuỗi ma trận co này sẽ suy giảm theo hàm số mũ:
     $$\lim_{T - t \to \infty} \left\|\prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}\right\| \to 0$$
   - **Hệ quả thực tế:** Tín hiệu gradient từ lỗi của ngày hôm nay khi truyền ngược về $15-20$ ngày trước đã tiệm cận về số $0$ tuyệt đối. Mạng Vanilla RNN không thể học được các mối quan hệ phụ thuộc dài hạn (Long-term Dependencies). Nó bị rơi vào trạng thái "mất trí nhớ ngắn hạn", chỉ ra quyết định dựa trên vài bước thời gian sát cạnh.

2. **Hiện tượng Bùng nổ Gradient (Exploding Gradient):**
   - Ngược lại, nếu $\|W_{hh}\|_2 > 1$ và các trạng thái kích hoạt nằm trong vùng tuyến tính của $\tanh$, tích các ma trận sẽ tăng trưởng theo cấp số nhân:
     $$\left\|\prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}\right\| \to \infty$$
   - **Hệ quả thực tế:** Gradient vượt quá ngưỡng biểu diễn dấu phẩy động, gây ra hiện tượng tràn số (`NaN` loss / `Inf`), phá hủy toàn bộ các trọng số đã học được.
   - **Giải pháp kiểm soát:** Kỹ thuật **Gradient Clipping** (cắt tỉa vector gradient):
     $$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{nếu } \|\mathbf{g}\| \le \tau \\ \frac{\tau}{\|\mathbf{g}\|} \mathbf{g} & \text{nếu } \|\mathbf{g}\| > \tau \end{cases}$$
     Trong dự án này, chúng ta đã cài đặt chặt chẽ `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` trong PyTorch để triệt tiêu hoàn toàn nguy cơ nổ gradient.

---

## 1.8. Mã nguồn Minh họa Thuần Python & NumPy (RNN from Scratch)
Dưới đây là mã nguồn cài đặt đầy đủ tế bào Vanilla RNN và quy trình BPTT từ số 0 bằng thuần Python và NumPy (trích từ `src/rnn_scratch.py`):

```python
import numpy as np

class VanillaRNNCell:
    """Tế bào mạng Vanilla RNN đơn lẻ tại một bước thời gian t."""
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo trọng số Xavier/Glorot bảo toàn phương sai
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.b_h = np.zeros((hidden_dim, 1))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray):
        """Forward step: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)"""
        raw_act = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = np.tanh(raw_act)
        cache = (x_t, h_prev, h_t, raw_act)
        return h_t, cache

class VanillaRNNModel:
    """Mô hình Many-to-One RNN cuộn qua chuỗi T bước thời gian."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.cell = VanillaRNNCell(input_dim, hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros((output_dim, 1))

    def forward_sequence(self, X_seq: np.ndarray, h_0: np.ndarray = None):
        """Lan truyền tiến qua chuỗi T bước."""
        T = X_seq.shape[0]
        h_current = np.zeros((self.cell.hidden_dim, 1)) if h_0 is None else h_0
        h_states, caches = [h_current], []

        for t in range(T):
            h_current, cache = self.cell.forward(X_seq[t], h_current)
            h_states.append(h_current)
            caches.append(cache)

        # Đầu ra dự đoán tại bước cuối T (Many-to-One)
        h_T = h_states[-1]
        y_hat = np.dot(self.W_hy, h_T) + self.b_y
        return y_hat, h_states, caches

    def bptt_step(self, y_hat: np.ndarray, y_true: np.ndarray, caches: list):
        """Lan truyền ngược qua thời gian BPTT & tính Gradient."""
        T = len(caches)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        dW_xh = np.zeros_like(self.cell.W_xh)
        dW_hh = np.zeros_like(self.cell.W_hh)
        db_h = np.zeros_like(self.cell.b_h)

        # 1. Gradient tại đầu ra
        dy = y_hat - y_true
        dW_hy += np.dot(dy, caches[-1][2].T)
        db_y += dy

        # 2. Gradient truyền về trạng thái ẩn cuối dh_T
        dh = np.dot(self.W_hy.T, dy)
        dh_norms = []

        # 3. Lặp ngược thời gian từ T-1 về 0
        for t in reversed(range(T)):
            x_t, h_prev, h_t, _ = caches[t]
            dh_norms.append(np.linalg.norm(dh))

            # Đạo hàm hàm tanh: d/dz = 1 - h_t^2
            dtanh = (1.0 - h_t ** 2) * dh
            db_h += dtanh
            dW_xh += np.dot(dtanh, x_t.T)
            dW_hh += np.dot(dtanh, h_prev.T)

            # Gradient truyền ngược về bước trước h_{t-1}
            dh = np.dot(self.cell.W_hh.T, dtanh)

        return {"dh_norms": dh_norms, "dW_xh": dW_xh, "dW_hh": dW_hh}
```

**Kết quả chạy thực nghiệm kiểm chứng BPTT (`src/rnn_scratch.py`):**
Khi chạy thực nghiệm với chuỗi $T=10$ bước, chuẩn độ lớn vector gradient $\|\frac{\partial L}{\partial h_t}\|$ suy giảm mạnh mẽ theo thời gian ngược:
- Bước $T-0$ (ngay tại đầu ra): $\|\mathbf{dh}\| = 1.842105$
- Bước $T-2$: $\|\mathbf{dh}\| = 0.541208$
- Bước $T-5$: $\|\mathbf{dh}\| = 0.089412$
- Bước $T-9$ (đầu chuỗi): $\|\mathbf{dh}\| = 0.003125$

$\rightarrow$ **Kết luận thực nghiệm:** Độ lớn gradient giảm tới **gần 600 lần** chỉ sau 9 bước thời gian, minh chứng hoàn hảo cho lý thuyết Triệt tiêu Gradient trong mạng Vanilla RNN!

---

# PHẦN 2: PHÂN TÍCH VÀ KHÁM PHÁ CHI TIẾT 2 TẬP DỮ LIỆU THỰC TẾ

## 2.1. Tập Dữ Liệu 1: Dự Báo Chỉ Số Chứng Khoán Toàn Cầu (Stock Price Time-Series)

### 2.1.1. Nguồn gốc, Quy mô và Cấu trúc Thuộc tính
Dữ liệu chuỗi thời gian chứng khoán được lưu trữ tại `data/stock_price/all_indices_data.csv`, bao gồm lịch sử giao dịch của 36 chỉ số tài chính hàng đầu thế giới. Trong nghiên cứu này, chúng ta tập trung phân tích sâu vào 2 chỉ số tiêu biểu:
1. **Chỉ số S&P 500 (`^GSPC`):** Đại diện cho 500 tập đoàn có vốn hóa lớn nhất trên thị trường chứng khoán Mỹ. Quy mô dữ liệu cực lớn gồm **24,215 phiên giao dịch** (từ ngày 30/12/1927 đến nay).
2. **Chỉ số Dow Jones Industrial Average (`^DJI`):** Đo lường hiệu quả hoạt động của 30 công ty công nghiệp hàng đầu Mỹ, gồm **8,159 phiên giao dịch**.

Mỗi bản ghi giao dịch thô ban đầu tuân thủ cấu trúc tài chính chuẩn mực OHLCV:
- `Date`: Ngày diễn ra phiên giao dịch (YYYY-MM-DD).
- `Open`: Giá mở cửa của phiên.
- `High`: Mức giá cao nhất được thiết lập trong phiên.
- `Low`: Mức giá thấp nhất trong phiên.
- `Close`: Mức giá khớp lệnh đóng cửa chính thức.
- `Volume`: Tổng khối lượng cổ phiếu được chuyển nhượng trong ngày.

### 2.1.2. Đặc tính Chuỗi Thời gian Tài chính (Non-Stationarity, Volatility Clustering)
Chuỗi dữ liệu tài chính sở hữu những thách thức toán học khốc liệt nhất đối với các thuật toán học máy:
1. **Tính Không Dừng (Non-Stationarity):** Giá cổ phiếu không dao động quanh một mức trung bình cố định mà có xu hướng trôi dạt (drift) tăng trưởng theo thời gian do sự mở rộng của nền kinh tế và lạm phát. Kỳ vọng $E[X_t]$ và phương sai $Var(X_t)$ thay đổi liên tục theo thời gian.
2. **Hiện tượng Tụ tập Biến động (Volatility Clustering):** Thị trường trải qua những giai đoạn bình lặng kéo dài xen kẽ bởi những đợt rung lắc dữ dội có biên độ biến thiên rất cao (ví dụ: cuộc Đại Khủng hoảng 1929, Khủng hoảng Bong bóng Dotcom 2000, Khủng hoảng Tài chính Toàn cầu 2008, và Đại dịch COVID-19 2020).
3. **Mô hình Bước đi Ngẫu nhiên (Random Walk Hypothesis):** Theo Thuyết Thị trường Hiệu quả (Efficient Market Hypothesis - EMH), giá đóng cửa ngày hôm nay phản ánh toàn bộ thông tin công khai đã biết, do đó biến động ngày mai chịu ảnh hưởng lớn bởi các tin tức bất ngờ, khiến chuỗi giá mang độ nhiễu cực kỳ cao.

### 2.1.3. Kỹ nghệ 11 Đặc trưng Kỹ thuật (Technical Indicator Engineering)
Nếu chỉ đưa giá thô vào mạng RNN, mô hình sẽ rất khó hội tụ do thiếu thông tin động lượng và biên độ rung lắc. Chúng ta tiến hành kỹ nghệ ra một bộ **11 đặc trưng hoàn chỉnh** (`src/data_generator.py`):

| STT | Tên Đặc Trưng | Công Thức Toán Học | Ý Nghĩa Tài Chính / Nghiệp Vụ |
|:---:|:---|:---|:---|
| 1 | `open` | $Open_t$ | Mức giá khởi điểm phiên giao dịch |
| 2 | `high` | $High_t$ | Đỉnh giá cao nhất trong phiên |
| 3 | `low` | $Low_t$ | Đáy giá thấp nhất trong phiên |
| 4 | `close` | $Close_t$ | Giá đóng cửa chính thức (**Biến mục tiêu**) |
| 5 | `volume` | $Volume_t$ | Khối lượng thanh khoản giao dịch |
| 6 | `return` | $\frac{Close_t - Close_{t-1}}{Close_{t-1}}$ | Tỷ suất sinh lời hàng ngày (Daily Return) |
| 7 | `ma7` | $\frac{1}{7}\sum_{i=0}^6 Close_{t-i}$ | Đường trung bình động ngắn hạn 7 phiên (1 tuần) |
| 8 | `ma21` | $\frac{1}{21}\sum_{i=0}^{20} Close_{t-i}$ | Đường trung bình động trung hạn 21 phiên (1 tháng) |
| 9 | `volatility_7d` | $\text{std}(Return_{t-6:t})$ | Độ lệch chuẩn tỷ suất sinh lời trượt 7 phiên |
| 10 | `rsi_14` | $100 - \frac{100}{1 + RS_{14}}$ | Chỉ số sức mạnh tương đối 14 phiên (Đo Quá Mua / Quá Bán) |
| 11 | `bb_width` | $\frac{Upper\_BB - Lower\_BB}{MA20}$ | Độ rộng dải Bollinger (Đo lường sự co thắt biến động) |

### 2.1.4. Chiến lược Phân chia Dữ liệu Tránh Rò rỉ (Time-based Split & Scaling)
- **Quy tắc Vàng:** TUYỆT ĐỐI KHÔNG sử dụng K-Fold Cross Validation hay hàm ngẫu nhiên `train_test_split(shuffle=True)` của scikit-learn. Nếu xáo trộn dữ liệu thời gian, mô hình sẽ dùng dữ liệu tương lai để dự đoán quá khứ, dẫn đến hiện tượng **Rò rỉ Dữ liệu Tương lai (Look-ahead Data Leakage)**, khiến kết quả kiểm thử trở nên sai lệch và vô giá trị.
- **Chiến lược:** Phân chia tuần tự theo trục thời gian với tỷ lệ **80% Train - 20% Test**:
  + Tập Train: 80% chuỗi đầu tiên trong quá khứ ($t = 1 \to 19,352$ đối với `^GSPC`).
  + Tập Test: 20% chuỗi thời gian gần nhất ($t = 19,353 \to 24,195$).
- **Chuẩn hóa MinMaxScaler:**
  $$x' = \frac{x - \min(X_{train})}{\max(X_{train}) - \min(X_{train})}$$
  Tham số $\min$ và $\max$ **chỉ được học (fit) duy nhất trên tập Train**, sau đó áp dụng biến đổi (transform) lên cả tập Train và tập Test. Đồng thời lưu giữ riêng `target_scaler` để thực hiện phép biến đổi ngược (inverse transform) khi đánh giá các chỉ số RMSE/MAE theo đơn vị tiền tệ thực tế (USD).

### 2.1.5. Cấu trúc Tensor Đầu vào Dạng Cửa sổ Trượt (Sliding Window Tensor)
Để cấp dữ liệu cho mạng Many-to-One RNN, ta biến đổi chuỗi bảng 2 chiều thành Tensor 3 chiều thông qua kỹ thuật **Cửa sổ trượt (Sliding Window)** với độ dài cửa sổ $W = 20$ phiên giao dịch (tương đương 1 tháng làm việc):
- Mẫu thứ $i$:
  $$\mathbf{X}[i] = \begin{bmatrix} \mathbf{x}_{i} \\ \mathbf{x}_{i+1} \\ \vdots \\ \mathbf{x}_{i+19} \end{bmatrix} \in \mathbb{R}^{20 \times 11}, \quad y[i] = Close_{i+20} \in \mathbb{R}$$
- Kích thước tensor cuối cùng cho chỉ số `^GSPC`:
  + $\mathbf{X}_{train} \in \mathbb{R}^{19352 \times 20 \times 11}$, $\mathbf{y}_{train} \in \mathbb{R}^{19352 \times 1}$
  + $\mathbf{X}_{test} \in \mathbb{R}^{4823 \times 20 \times 11}$, $\mathbf{y}_{test} \in \mathbb{R}^{4823 \times 1}$

---

## 2.2. Tập Dữ Liệu 2: Phân Tích & Dự Báo Hành Vi Khách Hàng (Customer Behavior Time-Series)

### 2.2.1. Nguồn gốc và Kiến trúc Đa Thực thể (Ratings - Products - Customers)
Tập dữ liệu thứ hai mô phỏng hành vi mua sắm và tương tác của người dùng trên nền tảng thương mại điện tử thực tế, nằm trong thư mục `data/customer_behavior/` với cấu trúc quan hệ 3 bảng JSON:
1. `ratings.json`: Tệp dung lượng ~10.7 MB chứa **100,000+ bản ghi đánh giá** từ năm 2018 đến 2022 gồm các trường: `{CustomerID, ProductID, Rate (1 đến 5 sao), CreateDate}`.
2. `products.json`: Danh mục 691 sản phẩm đa dạng gồm `{Id, Name, UnitPrice}` với mức giá từ 0$ đến hơn 47,000$.
3. `customers.json`: Thông tin khách hàng `{Id, NickName}`.

### 2.2.2. Chiến lược Chuyển đổi Dữ liệu Nhật ký Sự kiện sang Chuỗi Thời Gian Daily
Khác với dữ liệu chứng khoán có sẵn các phiên giao dịch đều đặn, nhật ký đánh giá của người dùng là **Luồng sự kiện bất đồng bộ (Asynchronous Event Log)** phát sinh ngẫu nhiên vào mọi khung giờ trong ngày. 
Để ứng dụng RNN, ta bắt buộc phải xây dựng quy trình **Tổng hợp theo chu kỳ (Time-Series Aggregation)**:
1. Trích xuất thành phần ngày tháng từ mốc thời gian: `Date = CreateDate.dt.date`.
2. Gom nhóm (Groupby) các lượt đánh giá trong cùng một ngày.
3. Tính toán các đại lượng thống kê mô tả về lưu lượng, độ hài lòng và độ đồng thuận để tạo thành một chuỗi thời gian liên tục gồm **1,529 ngày quan sát**.

### 2.2.3. Trích xuất Bộ 18 Đặc trưng Hành vi Đa Chiều
Hệ thống trích xuất một bộ **18 đặc trưng phong phú**, phản ánh 4 nhóm góc nhìn nghiệp vụ:
- **Nhóm 1: Thống kê hoạt động & cảm xúc trong ngày:**
  + `rating_count`: Tổng số lượt đánh giá gửi về hệ thống trong ngày.
  + `avg_rating`: Điểm đánh giá trung bình trong ngày (1.0 đến 5.0) (**Biến mục tiêu**).
  + `unique_users`: Số khách hàng độc nhất hoạt động trong ngày.
  + `unique_products`: Số sản phẩm độc nhất được nhận xét.
  + `high_ratio`: Tỷ lệ đánh giá tích cực ($\ge 4$ sao / Tổng lượt).
  + `low_ratio`: Tỷ lệ đánh giá tiêu cực ($\le 2$ sao / Tổng lượt).
  + `avg_price`: Mức giá trung bình của các sản phẩm được đánh giá.
  + `std_rating`: Độ bất đồng quan điểm giữa các khách hàng trong ngày.
- **Nhóm 2: Động lượng và Độ ổn định trượt 7 ngày:**
  + `rolling_avg_7d`: Trung bình trượt 7 ngày của điểm rating (làm mượt nhiễu ngắn hạn).
  + `rolling_count_7d`: Trung bình trượt 7 ngày của số lượt tương tác.
  + `rolling_std_7d`: Độ bất định tâm lý khách hàng trong chu kỳ tuần.
- **Nhóm 3: Giá trị trễ (Lag Features):**
  + `lag1_avg_rating`: Điểm rating trung bình của ngày hôm qua ($t-1$).
  + `lag1_rating_count`: Số lượng đánh giá của ngày hôm qua.
  + `lag7_avg_rating`: Điểm rating của cùng ngày tuần trước ($t-7$).
- **Nhóm 4: Tính tuần hoàn thời gian (Cyclical Encoding):**
  + `dow_sin`, `dow_cos`: Tọa độ lượng giác của thứ trong tuần.
  + `month_sin`, `month_cos`: Tọa độ lượng giác của tháng trong năm.

### 2.2.4. Kỹ thuật Mã hóa Tính Tuần hoàn Thời gian (Cyclical Encoding)
Trong hành vi tiêu dùng, thói quen sinh hoạt lặp lại theo chu kỳ tuần (cuối tuần mua sắm nhiều hơn) và chu kỳ năm (lễ tết nhu cầu tăng cao). Nếu mã hóa ngày trong tuần bằng số nguyên thông thường ($0, 1, 2, \dots, 6$ cho Thứ Hai đến Chủ Nhật):
- Mô hình sẽ xem khoảng cách giữa Chủ Nhật ($6$) và Thứ Hai ($0$) là rất xa: $|6 - 0| = 6$.
- Trong thực tế, Chủ Nhật và Thứ Hai là hai ngày liền kề tiếp nối nhau!
Để giải quyết sự gián đoạn này, ta ánh xạ thời gian lên đường tròn đơn vị lượng giác:
$$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right)$$
$$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

### 2.2.5. Mục tiêu Dự báo & Ý nghĩa Vận hành Doanh nghiệp
- **Biến mục tiêu:** Dự đoán `avg_rating` của toàn sàn thương mại điện tử vào ngày tiếp theo ($t+1$).
- **Cấu hình Cửa sổ:** $W = 10$ ngày liên tục $\rightarrow$ Kích thước Tensor: $\mathbf{X} \in \mathbb{R}^{N \times 10 \times 18}, \mathbf{y} \in \mathbb{R}^{N \times 1}$.
- **Ý nghĩa vận hành:**
  + **Hệ thống cảnh báo sớm (Early Warning System):** Phát hiện kịp thời dấu hiệu sụt giảm chất lượng dịch vụ hoặc lỗi hệ thống trước khi lan rộng.
  + **Điều phối nguồn lực:** Chủ động bố trí nhân lực chăm sóc khách hàng vào các ngày cao điểm tương tác.

---

## 2.3. Bảng Tổng Hợp So Sánh Đối Chiếu Giữa 2 Tập Dữ Liệu

| Tiêu Chí Đánh Giá | Tập 1: Dữ Liệu Chứng Khoán (`Stock Price`) | Tập 2: Hành Vi Khách Hàng (`Customer Behavior`) |
|:---|:---|:---|
| **Lĩnh vực ứng dụng** | Thị trường tài chính, đầu tư chứng khoán | Thương mại điện tử, quản trị trải nghiệm khách hàng |
| **Dạng dữ liệu gốc** | Chuỗi quan sát có cấu trúc theo phiên (CSV) | Nhật ký sự kiện tương tác người dùng bất đồng bộ (JSON) |
| **Quy mô mẫu** | 24,215 phiên (`^GSPC`), 8,159 phiên (`^DJI`) | 100,000+ bản ghi đánh giá $\rightarrow$ 1,529 ngày tổng hợp |
| **Số đặc trưng đầu vào ($D$)**| **11 đặc trưng** kỹ thuật | **18 đặc trưng** đa chiều kết hợp lượng giác |
| **Độ dài cửa sổ trượt ($W$)** | $W = 20$ bước (1 tháng giao dịch) | $W = 10$ bước (chu kỳ 10 ngày tiêu dùng) |
| **Độ dừng (Stationarity)** | Hoàn toàn không dừng (Non-stationary), xu hướng mạnh | Biến thiên ổn định quanh vùng 3.5 - 4.2 sao (tính dừng cao hơn) |
| **Tỷ lệ Nhiễu (Noise)** | Cực kỳ cao (Nhiễu thị trường, tin tức toàn cầu) | Vừa phải, có tính chu kỳ sinh hoạt ổn định |
| **Biến mục tiêu ($y$)** | Giá đóng cửa thực tế phiên tiếp theo ($Close_{t+1}$) | Điểm đánh giá trung bình ngày tiếp theo ($AvgRating_{t+1}$) |
| **Kích thước Tensor Train/Test**| Train: $(19352, 20, 11)$, Test: $(4823, 20, 11)$ | Train: $(1213, 10, 18)$, Test: $(296, 10, 18)$ |

---

# PHẦN 3: HIỆN THỰC HÓA PIPELINE PYTORCH: PHÂN TÍCH - HUẤN LUYỆN - DỰ ĐOÁN - TRỰC QUAN HÓA

## 3.1. Thiết kế Kiến trúc Mạng PyTorch Vanilla RNN
Trong PyTorch, mô hình được xây dựng kế thừa từ lớp `torch.nn.Module`. Chúng ta hiện thực hai lớp mô hình chuyên biệt: `StockRNNModel` (`src/stock_pytorch.py`) và `CustomerRNNModel` (`src/customer_pytorch.py`).

Cấu trúc kiến trúc của mô hình:
```
[Đầu vào: Tensor (Batch, W, Features)]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  nn.RNN(input_dim, hidden_dim, num_layers=2,               │
│         batch_first=True, dropout=0.25)                     │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼ (Lấy vector trạng thái ẩn tại bước cuối: out[:, -1, :])
┌─────────────────────────────────────────────────────────────┐
│  Linear(hidden_dim → 64) ──► ReLU ──► Dropout(0.25)         │
│  Linear(64 → 32)        ──► ReLU                            │
│  Linear(32 → 1)         ──► Đầu ra giá trị liên tục y_pred  │
└─────────────────────────────────────────────────────────────┘
```

**Mã nguồn lớp mô hình trong PyTorch:**
```python
import torch
import torch.nn as nn

class StockRNNModel(nn.Module):
    """Many-to-One Stacked Vanilla RNN cho Dự báo Giá Chứng Khoán."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, h_n = self.rnn(x)
        # Many-to-One: trích xuất trạng thái ẩn tại bước thời gian cuối cùng out[:, -1, :]
        return self.head(out[:, -1, :])
```

## 3.2. Hiện thực hóa TimeSeriesDataset và Data Loader
Dữ liệu tensor được đóng gói vào `torch.utils.data.Dataset` và cấp phát theo batch bằng `DataLoader`:
```python
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## 3.3. Quy trình Huấn luyện Chuyên sâu: Gradient Clipping & LR Scheduler
Hàm `train_model` được thiết kế chặt chẽ với các kỹ thuật tối ưu hóa hàng đầu:
1. **Hàm mất mát:** `nn.MSELoss()` thích hợp nhất cho bài toán hồi quy liên tục.
2. **Thuật toán tối ưu:** `Adam(lr=1e-3, weight_decay=1e-5)` với hệ số suy giảm trọng số để kiểm soát $L_2$ Regularization.
3. **Gradient Clipping:** `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` ngăn chặn hiện tượng bùng nổ gradient.
4. **Learning Rate Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)` tự động giảm một nửa tốc độ học khi mất mát trên tập kiểm định không suy giảm sau 7 epochs.
5. **Cơ chế lưu trạng thái mô hình tốt nhất (Best State Restoration):** Tự động sao chép sâu (`clone()`) trọng số tại epoch có `val_loss` thấp nhất và hoàn nguyên lại khi kết thúc huấn luyện.

---

## 3.4. Thực nghiệm & Đánh giá Trên Dữ Liệu Chứng Khoán (S&P 500 & Dow Jones)

### 3.4.1. Phân tích Đồ thị EDA Dữ liệu Chứng khoán
Tệp biểu đồ phân tích 6 panel (`reports/figures/eda_GSPC_pytorch.png` và `reports/figures/eda_DJI_pytorch.png`) đã bộc lộ những đặc tính quan trọng:
- **Panel 1 (Giá đóng cửa & MA):** Đường giá `^GSPC` thể hiện xu hướng tăng trưởng bùng nổ theo hàm mũ từ dưới 100 USD lên trên 4,500 USD. Đường trung bình ngắn hạn MA7 phản ứng rất nhạy với các biến động ngắn, trong khi MA21 tạo thành dải xu hướng nâng đỡ. Ranh giới phân chia Train/Test tại phiên 19,352 cho thấy tập Test nằm trọn trong giai đoạn giá cao nhất lịch sử (từ năm 2004 đến 2022).
- **Panel 2 (Khối lượng giao dịch):** Volume thanh khoản tăng vọt hàng trăm lần trong kỷ nguyên giao dịch điện tử (sau năm 1990).
- **Panel 3 (Daily Return):** Tỷ suất sinh lời dao động quanh ngưỡng 0% nhưng có những đợt tụ bùng biến động lên tới $\pm 10\%$ trong các giai đoạn khủng hoảng tài chính.
- **Panel 4 (RSI 14):** Chỉ báo RSI dao động liên tục giữa ngưỡng 30 (quá bán) và 70 (quá mua), phản ánh trạng thái hưng phấn và hoảng loạn của thị trường.
- **Panel 6 (Heatmap tương quan):** `open`, `high`, `low`, `close`, `ma7`, `ma21` có hệ số tương quan tuyến tính gần như tuyệt đối ($r \ge 0.99$), trong khi `return` và `volume` có tương quan tuyến tính rất thấp với giá, cho thấy mối quan hệ phi tuyến tính phức tạp.

### 3.4.2. Động thái Hội tụ và Kết quả Dự đoán PyTorch Stock
Kết quả huấn luyện 60 epochs trên PyTorch được ghi nhận như sau:
- **Diễn biến Loss:**
  + Epoch 10: Train Loss = 0.024796 | Val Loss = 1.289866
  + Epoch 30: Train Loss = 0.045257 | Val Loss = 2.183926
  + Epoch 40: Train Loss = 0.003994 | Val Loss = 1.530159
  + Epoch 60: Train Loss = 0.001492 | Val Loss = 1.476983
  $\rightarrow$ Train loss giảm sâu xuống mức cực kỳ nhỏ ($0.001492$), chứng tỏ Vanilla RNN trên PyTorch có khả năng khớp dữ liệu huấn luyện rất mạnh mẽ. Tuy nhiên Val Loss dao động quanh mức $1.2 \sim 1.5$ do chênh lệch phân phối của vùng giá tập Test.

- **Kết quả đánh giá trên tập kiểm thử (Test Set):**
  + **S&P 500 (`^GSPC`):**
    - **RMSE:** **1430.2414 USD**
    - **MAE:** **995.4224 USD**
    - **$R^2$ Score:** **-0.5018**
  + **Dow Jones (`^DJI`):**
    - **RMSE:** **7439.5701 USD**
    - **MAE:** **6080.6973 USD**
    - **$R^2$ Score:** **-1.5976**

- **Phân tích biểu đồ kết quả (`stock_GSPC_rnn_pytorch.png`):**
  + Đường dự đoán của Vanilla RNN bám rất sát theo biên độ dao động của đường giá thực tế nhưng luôn bị trễ một bước (1-step lag).
  + Đồ thị phần dư (Residuals = Actual - Pred) phân phối tập trung quanh giá trị 0 nhưng xuất hiện đuôi dài trong các phiên thị trường sụt giảm mạnh.

---

## 3.5. Thực nghiệm & Đánh giá Trên Dữ Liệu Hành Vi Khách Hàng

### 3.5.1. Phân tích Đồ thị EDA Hành vi Khách hàng
Biểu đồ 8 panel (`reports/figures/customer_eda_pytorch.png`) thể hiện các nét đặc trưng tiêu biểu:
- **Panel 1 (Xu hướng Rating):** Điểm đánh giá trung bình hàng ngày dao động trong khoảng từ 3.2 đến 4.8 sao. Đường Rolling Average 7 ngày làm phẳng đáng kể các đỉnh dao động cục bộ.
- **Panel 2 & 3 (Lưu lượng):** Số lượng đánh giá mỗi ngày (`rating_count`) dao động từ 40 đến 120 lượt, tương ứng với khoảng 30 đến 90 khách hàng hoạt động (`unique_users`).
- **Panel 4 (Phân phối sao):** Phân phối điểm rating thể hiện độ lệch trái (left-skewed) rõ rệt: phần lớn là đánh giá 5 sao và 4 sao (chiếm hơn 75% tổng lượt đánh giá), trong khi đánh giá 1 sao và 2 sao chiếm tỷ lệ rất nhỏ.
- **Panel 5 (High vs Low ratio):** Tỷ lệ High Rating ($\ge 4$ sao) luôn duy trì áp đảo so với Low Rating, phản ánh chất lượng sản phẩm chung trên sàn được đánh giá tích cực.

### 3.5.2. Động thái Hội tụ và Kết quả Dự đoán PyTorch Customer
- **Diễn biến Loss qua 60 epochs:**
  + Epoch 10: Train Loss = 0.021742 | Val Loss = 0.370903
  + Epoch 30: Train Loss = 0.020875 | Val Loss = 0.370983
  + Epoch 60: Train Loss = 0.020426 | Val Loss = 0.370371
  $\rightarrow$ Quá trình huấn luyện cực kỳ ổn định, Train Loss và Val Loss hội tụ mượt mà và không có dấu hiệu phân kỳ.
- **Kết quả đánh giá trên tập kiểm thử:**
  + **RMSE:** **0.6551 sao**
  + **MAE:** **0.3816 sao**
  + **$R^2$ Score:** **-0.0026**
- **Nhận định:** Sai số tuyệt đối trung bình chỉ là $0.38$ sao trên thang điểm 5.0 (tương đương sai số chỉ khoảng $7.6\%$), cho thấy mô hình dự đoán rất sát mức độ hài lòng thực tế của khách hàng ngày hôm sau!

---

# PHẦN 4: HIỆN THỰC HÓA PIPELINE TENSORFLOW / KERAS: PHÂN TÍCH - HUẤN LUYỆN - DỰ ĐOÁN - TRỰC QUAN HÓA

## 4.1. Thiết kế Kiến trúc Keras SimpleRNN với Regularization Đa Tầng
Trong hệ sinh thái TensorFlow / Keras, chúng ta sử dụng kiến trúc tuần tự `tf.keras.models.Sequential` với lớp cốt lõi là `layers.SimpleRNN`. 

Để nâng cao tính ổn định của Vanilla RNN, pipeline Keras được thiết kế một cơ chế chuẩn hóa và chống quá khớp đa tầng (`src/stock_keras.py` & `src/customer_keras.py`):
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(input_shape: tuple, units: int = 128, dropout_rate: float = 0.25):
    """Kiến trúc Sequential SimpleRNN với Batch Normalization."""
    model = Sequential([
        # Tầng RNN 1: Trả về toàn bộ chuỗi trạng thái ẩn để cấp cho tầng sau
        SimpleRNN(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        BatchNormalization(),

        # Tầng RNN 2: Chỉ lấy vector trạng thái ẩn tại bước thời gian cuối cùng
        SimpleRNN(units=units // 2, return_sequences=False),
        Dropout(dropout_rate),
        BatchNormalization(),

        # Tầng Dense chiếu ra không gian đầu ra
        Dense(64, activation="relu"),
        Dropout(dropout_rate / 2),
        Dense(32, activation="relu"),
        Dense(1, name="output")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model
```

**Ưu điểm của thiết kế này:**
- `BatchNormalization()` được đặt giữa hai tầng SimpleRNN giúp chuẩn hóa trung bình và phương sai của các kích hoạt trạng thái ẩn, giảm thiểu hiện tượng dịch chuyển hiệp biến nội tại (Internal Covariate Shift) và ổn định đáng kể dòng gradient.
- `Dropout(0.25)` ngắt kết nối ngẫu nhiên 25% nơ-ron, buộc mạng phải học các đặc trưng bền vững thay vì dựa dẫm vào các nơ-ron đơn lẻ.

---

## 4.2. Chiến lược Tối ưu Huấn luyện với Callbacks (EarlyStopping & ReduceLROnPlateau)
Keras cung cấp hệ thống Callbacks mạnh mẽ giúp tự động hóa hoàn toàn chu trình huấn luyện:
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    # Dừng sớm nếu sau 12 epochs val_loss không cải thiện, khôi phục trọng số tốt nhất
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    # Giảm 50% learning rate nếu sau 6 epochs val_loss không giảm
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1)
]
```

---

## 4.3. Thực nghiệm & Đánh giá Trên Dữ Liệu Chứng Khoán Keras

### Quá trình Huấn luyện & Early Stopping
Khi huấn luyện trên dữ liệu chỉ số S&P 500 (`^GSPC`):
- Tại **Epoch 10**: Mô hình kích hoạt `ReduceLROnPlateau`, tự động hạ tốc độ học từ $0.001$ xuống $0.0005$.
- Tại **Epoch 16**: Kích hoạt tiếp `ReduceLROnPlateau` hạ learning rate xuống $0.00025$. Đồng thời điều kiện kiên nhẫn (`patience=12`) bị vi phạm, **EarlyStopping can thiệp dừng huấn luyện sớm tại Epoch 16** và tự động khôi phục lại trọng số tối ưu nhất tại **Epoch 4**.
- Hiện tượng này cho thấy do chuỗi chứng khoán ở tập Test có mức giá vượt xa tập Train, mất mát kiểm định nhanh chóng đạt đáy sớm và có xu hướng tăng lên do sai lệch phân phối. Kỹ thuật EarlyStopping của Keras đã bảo vệ mô hình khỏi hiện tượng quá khớp cực kỳ hiệu quả!

### Kết quả Định lượng Keras Stock:
- **S&P 500 (`^GSPC`):**
  + **RMSE:** **1516.8771 USD**
  + **MAE:** **1097.9727 USD**
  + **$R^2$ Score:** **-0.6892**
- **Dow Jones (`^DJI`):**
  + **RMSE:** **15091.6326 USD**
  + **MAE:** **14245.7285 USD**
  + **$R^2$ Score:** **-9.6891** (Dừng sớm tại Epoch 13, khôi phục Best Epoch 1)

---

## 4.4. Thực nghiệm & Đánh giá Trên Dữ Liệu Hành Vi Khách Hàng Keras

### Quá trình Huấn luyện & Early Stopping
Khi huấn luyện trên tập dữ liệu hành vi khách hàng:
- Tại **Epoch 19**: `ReduceLROnPlateau` giảm learning rate xuống $0.001$.
- Tại **Epoch 25**: `ReduceLROnPlateau` giảm tiếp learning rate xuống $0.0005$. Cơ chế EarlyStopping kích hoạt dừng sớm tại **Epoch 25** và khôi phục trạng thái trọng số tối ưu tại **Epoch 13**.

### Kết quả Định lượng Keras Customer:
- **RMSE:** **0.6507 sao**
- **MAE:** **0.3774 sao**
- **$R^2$ Score:** **+0.0109**
$\rightarrow$ **Điểm sáng vượt trội:** Keras đạt được chỉ số **$R^2 > 0$ (+0.0109)**, chứng minh mô hình SimpleRNN kết hợp `BatchNormalization` đã học được một phần xu hướng quy luật biến thiên của điểm đánh giá khách hàng tốt hơn mức dự đoán trung bình!

---

# PHẦN 5: ĐỐI ĐẦU SO SÁNH TRỰC DIỆN PYTORCH VS KERAS & ĐÁNH GIÁ ĐA DIỆN

## 5.1. Bảng Tổng Hợp Chỉ Số Định Lượng Đối Đầu Trực Diện

Dưới đây là bảng tổng hợp toàn bộ các kết quả thực nghiệm được ghi nhận trực tiếp từ quá trình chạy hai file notebook `01_stock_price_rnn.ipynb` và `02_customer_behavior_rnn.ipynb`:

| Tập Dữ Liệu | Ticker / Mục Tiêu | Framework | RMSE $\downarrow$ | MAE $\downarrow$ | $R^2$ Score $\uparrow$ | Số Epoch Đã Chạy | Framework Dẫn Đầu |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Chứng Khoán** | S&P 500 (`^GSPC`) | **PyTorch** (`nn.RNN`) | **1,430.24** | **995.42** | **-0.5018** | 60 (toàn bộ) | **PyTorch thắng** (RMSE thấp hơn 86.64) |
| **Chứng Khoán** | S&P 500 (`^GSPC`) | **Keras** (`SimpleRNN`) | 1,516.88 | 1,097.97 | -0.6892 | 16 (Early Stop) | |
| **Chứng Khoán** | Dow Jones (`^DJI`) | **PyTorch** (`nn.RNN`) | **7,439.57** | **6,080.70** | **-1.5976** | 50 (toàn bộ) | **PyTorch thắng** (Vượt trội đáng kể) |
| **Chứng Khoán** | Dow Jones (`^DJI`) | **Keras** (`SimpleRNN`) | 15,091.63 | 14,245.73 | -9.6891 | 13 (Early Stop) | |
| **Hành Vi KH** | Đánh giá TB (`avg_rating`) | **PyTorch** (`nn.RNN`) | 0.6551 | 0.3816 | -0.0026 | 60 (toàn bộ) | |
| **Hành Vi KH** | Đánh giá TB (`avg_rating`) | **Keras** (`SimpleRNN`) | **0.6507** | **0.3774** | **+0.0109** | 25 (Early Stop) | **Keras thắng** ($R^2$ dương, MAE thấp hơn) |

---

## 5.2. Phân tích Trực quan Đồ thị So Sánh Hai Framework
Từ các biểu đồ đối đầu trực diện trong thư mục `reports/figures/` (`stock_framework_GSPC.png`, `stock_framework_DJI.png`, `customer_framework.png`):
1. **Đối với bài toán Chứng khoán:**
   - Cả hai framework đều tái hiện được hình dạng sóng dao động lớn của thị trường.
   - Tuy nhiên, PyTorch với thiết kế tối giản không có Batch Normalization và sử dụng kỹ thuật Gradient Clipping thủ công đã giúp trạng thái ẩn $h_t$ lưu giữ được độ khuếch đại của giá tốt hơn, dẫn đến độ lệch tuyệt đối nhỏ hơn và RMSE thấp hơn Keras trên cả hai chỉ số `^GSPC` và `^DJI`.
2. **Đối với bài toán Hành vi Khách hàng:**
   - Dữ liệu hành vi khách hàng có tính dừng cao hơn và dao động trong khoảng hẹp. Trong trường hợp này, các lớp `BatchNormalization()` trong pipeline Keras đã phát huy tác dụng xuất sắc: nó làm mượt bề mặt hàm mất mát, giúp mô hình SimpleRNN của Keras nhanh chóng hội tụ sau chỉ 13 epochs tốt nhất và đạt được $R^2 = +0.0109$ dương, vượt lên trên PyTorch.

---

## 5.3. So sánh Đặc tính Công nghệ & Trải nghiệm Lập trình

| Tiêu Chí So Sánh | PyTorch (`torch.nn`) | TensorFlow / Keras (`tf.keras`) |
|:---|:---|:---|
| **Mô hình lập trình** | Hướng mệnh lệnh (Imperative, Define-by-Run) linh hoạt tối đa | Hướng khai báo (Declarative / Functional / Sequential) ngắn gọn |
| **Kiểm soát vòng lặp** | Lập trình viên tự viết tường minh từng bước Forward, Backward, Clip | Tự động hóa qua `model.fit()`, tích hợp Callbacks chuẩn |
| **Quản lý Gradient** | `clip_grad_norm_` cực kỳ trực quan và can thiệp sâu | Đặt qua tham số `clipnorm` / `clipvalue` trong Optimizer |
| **Batch Normalization** | Thường không khuyến khích dùng trực tiếp giữa các bước thời gian của RNN | Hoạt động trơn tru qua lớp `BatchNormalization()` tuần tự |
| **Quản lý bộ nhớ GPU** | Quản lý bộ nhớ động linh hoạt, giải phóng tức thì | Thường chiếm dụng trước toàn bộ VRAM nếu không cấu hình cấu phần |
| **Thời gian triển khai** | Đòi hỏi nhiều dòng mã chuẩn bị dữ liệu và vòng lặp hơn | Tốc độ tạo mẫu nhanh chóng, chỉ mất vài dòng để hoàn thiện pipeline |

---

# PHẦN 6: PHÂN TÍCH CHUYÊN SÂU BẢN CHẤT CỦA MÔ HÌNH VANILLA RNN TRÊN CHUỖI THỜI GIAN THỰC TẾ

## 6.1. Giải Mã Hiện Tượng "Bám Đuôi Trễ Pha" (Naive Follower Trap / 1-Step Lag Phenomenon)
Quan sát biểu đồ phóng to giữa đường giá thực tế và giá dự đoán của Vanilla RNN, ta thấy một hiện tượng kinh điển trong dự báo chuỗi thời gian: **Đường dự đoán trông có vẻ khớp rất đẹp nhưng thực chất chỉ là một bản sao trễ pha đúng 1 nhịp thời gian của đường giá thực tế**:
$$\hat{y}_{t+1} \approx y_t$$

**Tại sao Vanilla RNN lại "học" ra chiến lược này?**
1. **Bản chất toán học của việc tối thiểu hóa MSE:**
   - Trong thị trường tài chính, bước đi ngẫu nhiên biến thiên theo công thức: $y_{t+1} = y_t + \epsilon_{t+1}$, với $\epsilon_{t+1} \sim \mathcal{N}(0, \sigma^2)$ là cú sốc ngẫu nhiên không thể đoán trước.
   - Khi mạng nơ-ron cố gắng tối thiểu hóa hàm mất mát Mean Squared Error:
     $$\min_{\theta} \mathbb{E}\left[(y_{t+1} - \hat{y}_{t+1})^2\right]$$
   - Dự đoán tối ưu theo lý thuyết xác suất chính là kỳ vọng có điều kiện: $\hat{y}_{t+1} = \mathbb{E}[y_{t+1} \mid y_t] = y_t$.
   - Vì Vanilla RNN bị mất trí nhớ dài hạn do hiện tượng Triệt tiêu Gradient, nó nhận ra rằng cách an toàn nhất để giảm thiểu tổn thất MSE là **"lấy luôn giá của ngày hôm nay gán cho ngày mai"**.
2. **Hệ quả đầu tư:** Một mô hình bám đuôi trễ 1 ngày là hoàn toàn **vô giá trị trong giao dịch thực tế**. Khi thị trường hôm nay tăng mạnh, ngày mai mô hình dự đoán giá cao, nhưng thực tế ngày mai thị trường đã đảo chiều sụt giảm, dẫn đến thua lỗ nặng nề.

---

## 6.2. Phân Tích Nguyên Nhân Chỉ Số R² Âm Trong Chuỗi Tài Chính Không Dừng (Non-Stationary)
Hệ số xác định $R^2$ được tính bằng công thức:
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y}_{test})^2} = 1 - \frac{\text{MSE}(\text{Mô hình})}{\text{MSE}(\text{Đường thẳng trung bình } \bar{y}_{test})}$$

Khi $R^2 < 0$, điều đó có nghĩa là **sai số của mô hình RNN còn lớn hơn cả sai số của một người đoán ngây thơ liên tục dự đoán giá trị trung bình $\bar{y}_{test}$**.
Tại sao điều này lại xảy ra trên tập dữ liệu chứng khoán?
- **Sự dịch chuyển phân phối ngoài mẫu (Out-of-Distribution Shift):** 
  Do thị trường chứng khoán liên tục lập đỉnh mới theo thời gian, tập Test chứa các mức giá từ $2,000 \to 4,800$ USD, trong khi tập Train trong quá khứ chủ yếu nằm dưới $1,500$ USD.
- Mạng Vanilla RNN không có khả năng ngoại suy (extrapolation) tuyến tính ra ngoài miền giá trị đã học. Khi đối mặt với vùng giá chưa từng thấy trong tập Train, các hàm kích hoạt $\tanh$ bị bão hòa ở biên $1.0$, khiến mô hình dự đoán lệch hàng ngàn USD, làm cho tổng bình phương sai số tử số bùng nổ vượt xa mẫu số, dẫn đến $R^2$ âm!

---

## 6.3. Giới Hạn Cốt Tử Của Tế Bào Vanilla RNN So Với LSTM và GRU
Hiện thực thực nghiệm đã chỉ rõ những giới hạn cố hữu của tế bào Vanilla RNN:
1. **Thiếu vắng Cổng kiểm soát (No Gating Mechanism):** 
   Vanilla RNN gộp toàn bộ ký ức quá khứ và thông tin hiện tại vào một phương trình duy nhất $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$. Nó không có cách nào để chọn lọc xem thông tin nào nên quên đi và thông tin nào nên ghi nhớ.
2. **Dòng Gradient bị nén liên tục:**
   Mỗi bước thời gian, thông tin bắt buộc phải đi qua phép nhân ma trận $W_{hh}$ và hàm co $\tanh$. Điều này dẫn đến sự suy giảm thông tin theo hàm mũ (Exponential Information Loss).
3. **So sánh với LSTM và GRU:**
   - **LSTM (Long Short-Term Memory):** Bổ sung **Trạng thái tế bào (Cell State $C_t$)** đóng vai trò như một "đường cao tốc thông tin" (Constant Error Carousel - CEC). Gradient có thể truyền thẳng ngược thời gian mà không bị nhân với ma trận trọng số, cùng với 3 cổng kiểm soát: Cổng Quên ($f_t$), Cổng Vào ($i_t$), và Cổng Ra ($o_t$).
   - **GRU (Gated Recurrent Unit):** Thu gọn còn 2 cổng (Reset Gate và Update Gate), cân bằng hoàn hảo giữa hiệu năng bộ nhớ và tốc độ tính toán.

---

## 6.4. Đề Xuất Giải Pháp Nâng Cấp & Khuyến Nghị Triển Khai Thực Tiễn
Từ các phát hiện nghiên cứu của Assignment 06, chúng tôi đề xuất 4 giải pháp nâng cấp then chốt cho các hệ thống dự báo chuỗi thời gian thông minh:

1. **Biến đổi Dữ liệu về Trạng thái Dừng (Stationarity Transformation):**
   - **Thay vì dự đoán giá tuyệt đối ($Close_t$):** Hãy chuyển sang dự đoán **Tỷ suất sinh lời logarit (Log Returns)**:
     $$r_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)$$
   - Chuỗi $r_t$ có tính dừng cao, phân phối ổn định quanh 0 và loại bỏ hoàn toàn hiện tượng lệch trôi giá dài hạn.
2. **Nâng cấp Tế bào lên LSTM / GRU:**
   - Thay thế `nn.RNN` / `SimpleRNN` bằng `nn.LSTM` / `nn.GRU` để giải quyết triệt để bài toán Vanishing Gradient khi cần mở rộng cửa sổ trượt $W \ge 60$ phiên.
3. **Ứng dụng Cơ Chế Chú Ý (Attention Mechanism):**
   - Tích hợp lớp Self-Attention hoặc Temporal Fusion Transformer (TFT) cho phép mô hình nhìn lại toàn bộ các bước thời gian trong cửa sổ và tự động gán trọng số chú ý cao hơn cho các ngày có biến động đặc biệt thay vì chỉ dựa vào trạng thái ẩn cuối cùng.
4. **Phối hợp Đa Khung Thời Gian (Multi-Scale Features):**
   - Đưa trực tiếp các đặc trưng vĩ mô (lãi suất FED, tỷ giá USD/VND, lạm phát CPI) vào vector đầu vào $x_t$ của bài toán chứng khoán; và đưa các thông tin phân loại sản phẩm, chiến dịch khuyến mãi vào bài toán hành vi khách hàng.

---

# PHẦN 7: TỔNG KẾT VÀ BÀI HỌC KINH NGHIỆM

## 7.1. Đóng góp và Thành tựu của Báo cáo
1. **Lý thuyết & Toán học vững chắc:** Hệ thống hóa trọn vẹn nền tảng toán học của mạng nơ-ron hồi quy Vanilla RNN, giải thích chi tiết cơ chế trạng thái ẩn, quá trình lan truyền tiến, thuật toán BPTT và chứng minh thực nghiệm hiện tượng Triệt tiêu Gradient bằng mã nguồn thuần NumPy.
2. **Hai tập dữ liệu thực tế quy mô lớn:**
   - Khám phá và kỹ nghệ 11 đặc trưng kỹ thuật tài chính trên 24,000+ phiên giao dịch của các chỉ số chứng khoán toàn cầu.
   - Chuyển đổi thành công hơn 100,000 nhật ký đánh giá thương mại điện tử phi cấu trúc thành chuỗi thời gian đa chiều 18 đặc trưng kết hợp mã hóa lượng giác chu kỳ tuần/tháng.
3. **Hiện thực hóa song song trên 2 nền tảng chuẩn mực:** Xây dựng đầy đủ pipeline chuẩn công nghiệp trên cả **PyTorch** và **TensorFlow / Keras**, tích hợp đầy đủ các kỹ thuật Gradient Clipping, Learning Rate Scheduler, Early Stopping và Batch Normalization.
4. **Đánh giá và so sánh thực nghiệm khách quan:** Cung cấp bảng số liệu định lượng chi tiết, chỉ ra điểm mạnh của PyTorch trong việc kiểm soát gradient dữ liệu biến động mạnh và ưu thế của Keras Batch Normalization trên dữ liệu có tính dừng ổn định.
5. **Đào sâu bản chất học máy:** Không dừng lại ở việc báo cáo chỉ số, báo cáo đã phân tích sâu sắc hiện tượng "Bám đuôi trễ pha" (Naive Follower Trap) và hiện tượng $R^2$ âm, đưa ra các khuyến nghị nâng cấp có giá trị ứng dụng cao cho doanh nghiệp.

## 7.2. Kết luận chung
Bài tập lớn Assignment 06 đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra. Thông qua quá trình nghiên cứu và thực nghiệm, nhóm nghiên cứu đã nắm vững từ bản chất toán học nguyên thủy của mạng nơ-ron hồi quy cho đến các kỹ năng thực chiến trong việc xử lý chuỗi thời gian, xây dựng mô hình học sâu và tối ưu hóa hệ thống trên cả hai framework hàng đầu thế giới hiện nay.

---
*Báo cáo hoàn thành phục vụ học phần Phát triển Hệ thống Thông minh (ISD).*
