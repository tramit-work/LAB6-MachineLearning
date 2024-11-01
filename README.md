# LAB 6 - Machine Learning

## 1. Công nghệ sử dụng
- **Frameworks**: 
  - `torch`
  - `numpy`
  - `matplotlib`
  - `flask`

## 2. Thuật toán
**MLP (Multi-Layer Perceptron)** là một loại mạng nơ-ron nhân tạo, được sử dụng trong học sâu. Nó bao gồm nhiều lớp nơ-ron, bao gồm ít nhất một lớp ẩn giữa lớp đầu vào và lớp đầu ra. MLP là một mô hình học máy có giám sát, thường được áp dụng cho các bài toán phân loại và hồi quy.
**VD:**
<p align="center">
  <img src="https://aiml.com/wp-content/uploads/2022/06/Multilayer-perceptron-MLP.png" alt="What is a Multilayer Perceptron (MLP) or a Feedforward Neural Network (FNN)? - AIML.com" style="width: 70%; height: auto;">
<p align="right">
  <em>What is a Multilayer Perceptron (MLP) or a Feedforward Neural Network (FNN)? - AIML.com</em>
  
### Loss Functions

**Loss function** (hàm mất mát) là một thành phần quan trọng trong quá trình huấn luyện mô hình, giúp đo lường độ chính xác của dự đoán so với giá trị thực tế. Dưới đây là một số loại hàm mất mát phổ biến:

- **Cross Entropy Loss**:
  - Sử dụng trong các bài toán phân loại đa lớp. Nó đo lường sự khác biệt giữa phân phối xác suất thực tế và phân phối xác suất dự đoán.

- **Mean Square Error (MSE)**:
  - Sử dụng trong các bài toán hồi quy. Nó tính toán trung bình của bình phương sai số giữa giá trị dự đoán và giá trị thực tế. MSE nhấn mạnh các sai số lớn hơn.

- **Binary Cross Entropy Loss**:
  - Sử dụng cho các bài toán phân loại nhị phân. Nó đo lường sự khác biệt giữa hai phân phối xác suất cho các lớp nhị phân.
**VD:**
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*hC94sOK3bB4junpyRMnXwg.png" alt="Understanding Loss Functions for Classification | by Nghi Huynh | Medium" style="width: 70%; height: auto;">
<p align="right">
  <em>Understanding Loss Functions for Classification | by Nghi Huynh | Medium</em>
  
### 3. Activation Functions

**Activation function** (hàm kích hoạt) là một thành phần quan trọng trong mạng nơ-ron, quyết định đầu ra của mỗi nơ-ron. Dưới đây là một số hàm kích hoạt phổ biến:

- **Sigmoid**:
  - Hàm này chuyển đổi đầu vào thành giá trị trong khoảng (0, 1), thường được sử dụng trong các bài toán phân loại nhị phân.

- **ReLU (Rectified Linear Unit)**:
  - Hàm này trả về giá trị đầu vào nếu nó dương, ngược lại trả về 0. ReLU giúp tăng tốc độ huấn luyện bằng cách giảm thiểu vấn đề gradient vanishing.

- **Softmax**:
  - Hàm này chuyển đổi một vector thành một phân phối xác suất, thường được sử dụng trong lớp đầu ra của MLP cho các bài toán phân loại đa lớp.

- **Tanh (Hyperbolic Tangent)**:
  - Hàm này chuyển đổi đầu vào thành giá trị trong khoảng (-1, 1). Tanh thường được sử dụng trong các lớp ẩn để cải thiện khả năng học.
**VD:**
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*ZafDv3VUm60Eh10OeJu1vw.png" alt="Introduction to Different Activation Functions for Deep Learning | by Shruti Jadon | Medium" style="width: 70%; height: auto;">
<p align="right">
  <em>Introduction to Different Activation Functions for Deep Learning | by Shruti Jadon | Medium</em>
  
## 3. Hiển thị kết quả lên Website - MLP
<p align="center">
  <img src="https://github.com/tramit-work/LAB6-MachineLearning/blob/main/photos/mlp.mov" alt="Video MLP">
</p>

## 4. Đối với các bài toán có sự so sánh
<p align="center">
  <img src="https://github.com/tramit-work/LAB6-MachineLearning/blob/main/photos/formula.png" alt="So sánh">
</p>
