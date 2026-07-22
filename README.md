# Xây dựng mô hình dự đoán Mức độ béo phì ở cá nhân

## 📌 Tổng quan dự án  

Xây dựng một mô hình Phân loại mức độ béo phì ở các nhân. Nhằm giải đáp cho 3 câu hỏi chính:
- Mỗi quan hệ giữa khả năng gây bên tình trạng **béo phì** là gì?
- Lựa chọn ra các **đặc trưng ảnh hường** đến việc nhận biết béo phì.
- Sử dụng mô hình Học máy để dự đoán 1 cá nhân thuộc béo phì hay không?

Bên cạnh đó cũng nhằm xây dựng một pipeline Học máy có khả năng: 
- Huấn luyện so sánh **5 mô hình Học Máy**
- Kết hợp quản lý các tham só bằng **Hydra**

### 🎯 Mục tiêu 

| Objective                | Description                                  |
| ------------------------ | -------------------------------------------- |
| Reproducibility          | Có thể tái lập experiment bằng configuration |
| Experiment Tracking      | Theo dõi toàn bộ experiment bằng MLflow      |
| Configuration Management | Quản lý config tập trung với Hydra           |
| Model Comparison         | So sánh 5 mô hình trên cùng một tiêu chí     |
| Scalability              | Dễ dàng thêm model/config mới                |



**Lựa chọn bộ dữ liệu:** Chứa các đặc trưng có thể ảnh hưởng đến mức độ béo phì của một người, xuất phát từ các quốc gia như Mexico, Peru hay Columbia gồm 17 đặc trưng khác nhau. 77% dữ liệu trong tập này sử dụng công cụ Weka và lọc SMOTE, 23% dữ liệu được thu thập từ các trang web.
Các đặc trưng trong bộ dữ liệu xoay quanh các thông tin:
- Thông tin cá nhân
- Các thói quan sinh hoạt
- Mối quan tâm

Nguồn của bộ dữ liệu: [Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)

---

## ⚙️ Công nghệ sử dụng

| Technology           | Purpose                     |
| -------------------- | --------------------------- |
| Python               | Programming language        |
| Scikit-learn         | Machine Learning            |
| Hydra                | Configuration management    |
| Pandas               | Data processing             |
| NumPy                | Numerical computation       |
| Matplotlib / Seaborn | Visualization               |



