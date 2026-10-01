"""
ISD-Assignment06: RNN From Scratch (Pure Python & NumPy)
Minh họa trực quan và chính xác toán học của Vanilla RNN:
1. Hidden State Transition: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
2. Output Layer: y_hat_t = W_hy @ h_t + b_y
3. Lan truyền tiến qua chuỗi thời gian (Forward Pass)
4. Lan truyền ngược qua thời gian (Backpropagation Through Time - BPTT)
"""

import numpy as np

class VanillaRNNCell:
    """
    Một tế bào (cell) mạng nơ-ron hồi quy đơn giản tại một bước thời gian t.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo trọng số Xavier / Glorot
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.b_h = np.zeros((hidden_dim, 1))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray):
        """
        Thực hiện một bước tính trạng thái ẩn h_t:
        x_t: vector đầu vào tại bước t (input_dim, 1)
        h_prev: vector trạng thái ẩn từ bước t-1 (hidden_dim, 1)
        """
        # Linear combination: W_xh @ x_t + W_hh @ h_{t-1} + b_h
        raw_activation = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        
        # Phi tuyến hóa bằng hàm tanh
        h_t = np.tanh(raw_activation)
        
        # Lưu cache để dùng trong BPTT
        cache = (x_t, h_prev, h_t, raw_activation)
        return h_t, cache

class VanillaRNNModel:
    """
    Mô hình RNN tuần tự qua chuỗi T bước thời gian.
    Nhận đầu vào chuỗi X = [x_1, x_2, ..., x_T]
    Dự đoán đầu ra tại bước cuối cùng (Many-to-One) hoặc toàn bộ chuỗi (Many-to-Many).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.cell = VanillaRNNCell(input_dim, hidden_dim)
        
        # Trọng số tầng đầu ra (Projection Head)
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros((output_dim, 1))

    def forward_sequence(self, X_seq: np.ndarray, h_0: np.ndarray = None):
        """
        X_seq: mảng (T, input_dim, 1) biểu diễn chuỗi T bước
        h_0: trạng thái ẩn khởi đầu (hidden_dim, 1)
        """
        T = X_seq.shape[0]
        if h_0 is None:
            h_0 = np.zeros((self.hidden_dim, 1))
            
        h_states = [h_0]
        caches = []
        
        # Cuộn qua thời gian (Loop over time steps)
        h_current = h_0
        for t in range(T):
            x_t = X_seq[t]
            h_current, cache = self.cell.forward(x_t, h_current)
            h_states.append(h_current)
            caches.append(cache)
            
        # Dự đoán tại bước cuối cùng (Many-to-One): y_hat = W_hy @ h_T + b_y
        h_T = h_states[-1]
        y_hat = np.dot(self.W_hy, h_T) + self.b_y
        
        return y_hat, h_states, caches

    def compute_loss(self, y_hat: np.ndarray, y_true: np.ndarray):
        """
        Tính Mean Squared Error (MSE) cho bài toán hồi quy
        """
        loss = 0.5 * np.sum((y_hat - y_true) ** 2)
        return loss

    def bptt_step(self, y_hat: np.ndarray, y_true: np.ndarray, caches: list):
        """
        Minh họa Backpropagation Through Time (BPTT)
        Giải thích hiện tượng Vanishing Gradient:
        Gradient dL/dh_t được nhân liên tiếp qua ma trận W_hh^T và đạo hàm tanh (1 - h_t^2).
        """
        T = len(caches)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        dW_xh = np.zeros_like(self.cell.W_xh)
        dW_hh = np.zeros_like(self.cell.W_hh)
        db_h = np.zeros_like(self.cell.b_h)

        # Gradient tại tầng output
        dy = y_hat - y_true # (output_dim, 1)
        dW_hy += np.dot(dy, caches[-1][2].T) # dy @ h_T.T
        db_y += dy

        # Gradient truyền về trạng thái ẩn cuối cùng dh_T
        dh = np.dot(self.W_hy.T, dy) # (hidden_dim, 1)

        # BPTT qua từng bước thời gian ngược từ T-1 về 0
        dh_magnitudes = []
        for t in reversed(range(T)):
            x_t, h_prev, h_t, _ = caches[t]
            dh_magnitudes.append(np.linalg.norm(dh))

            # Đạo hàm qua hàm kích hoạt tanh: d/dz [tanh(z)] = 1 - tanh(z)^2 = 1 - h_t^2
            dtanh = (1.0 - h_t ** 2) * dh

            # Đạo hàm theo các trọng số và bias
            db_h += dtanh
            dW_xh += np.dot(dtanh, x_t.T)
            dW_hh += np.dot(dtanh, h_prev.T)

            # Gradient truyền ngược về bước thời gian trước đó (h_{t-1})
            # Đây chính là mấu chốt: phép nhân W_hh.T @ dtanh được lặp đi lặp lại T lần!
            dh = np.dot(self.cell.W_hh.T, dtanh)

        return {
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "dW_hy": dW_hy,
            "db_h": db_h,
            "db_y": db_y,
            "dh_magnitudes_reverse": dh_magnitudes
        }

if __name__ == "__main__":
    print("=== DEMO RUN: Vanilla RNN Step-by-Step ===")
    seq_len = 10
    in_dim = 4
    hidden_size = 8
    out_dim = 1

    model = VanillaRNNModel(input_dim=in_dim, hidden_dim=hidden_size, output_dim=out_dim)

    # Tạo một chuỗi dữ liệu giả lập (10 ngày, mỗi ngày 4 đặc trưng)
    X = np.random.randn(seq_len, in_dim, 1)
    y_target = np.array([[42.5]])

    # Lan truyền tiến (Forward)
    y_pred, h_states, caches = model.forward_sequence(X)
    loss = model.compute_loss(y_pred, y_target)

    print(f"1. Chiều dữ liệu đầu vào: {X.shape}")
    print(f"2. Trạng thái ẩn cuối cùng h_T:\n{h_states[-1].T}")
    print(f"3. Giá trị dự đoán y_pred: {y_pred[0, 0]:.4f} | Thực tế: {y_target[0, 0]:.4f}")
    print(f"4. MSE Loss: {loss:.4f}")

    # Lan truyền ngược qua thời gian (BPTT)
    grads = model.bptt_step(y_pred, y_target, caches)
    print("\n5. Độ lớn Gradient theo thời gian (Từ T ngược về 0):")
    for step, norm in enumerate(grads["dh_magnitudes_reverse"]):
        print(f"   Bước T-{step} -> norm(dh) = {norm:.6f}")
    print("-> Nhận xét: Gradient suy giảm dần (Vanishing Gradient) khi lan truyền ngược qua chuỗi dài!")
