![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Academic%20Project-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Hydra](https://img.shields.io/badge/Hydra-Configuration%20Management-purple)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)

# 🩺 Obesity Level Classification – Machine Learning Project

## 📌 Giới thiệu

Béo phì là một vấn đề sức khỏe có liên quan đến nhiều yếu tố như đặc điểm nhân khẩu học,
thói quen ăn uống, mức độ hoạt động thể chất và lối sống.

Trong bối cảnh đó, việc phân tích dữ liệu và áp dụng các thuật toán Machine Learning có thể
hỗ trợ khám phá mối quan hệ giữa các đặc trưng cá nhân và mức độ béo phì.

Đề tài này áp dụng quy trình **Machine Learning** để phân tích dữ liệu và xây dựng các mô hình
phân loại mức độ béo phì của một cá nhân dựa trên các đặc trưng về nhân khẩu học,
thói quen ăn uống và lối sống.

Project tập trung vào việc:

- Phân tích các yếu tố có liên quan đến mức độ béo phì
- Khám phá và lựa chọn các đặc trưng quan trọng
- Xây dựng mô hình Machine Learning để phân loại mức độ béo phì
- Thực nghiệm và so sánh 5 thuật toán Machine Learning
- Sử dụng Hydra để quản lý configuration và hyperparameters
- Sử dụng MLflow để theo dõi và quản lý các experiment

> ⚠️ **Lưu ý:** Project phục vụ mục đích học tập và nghiên cứu Machine Learning.
> Kết quả mô hình không được sử dụng để thay thế cho chẩn đoán hoặc đánh giá y tế chuyên môn.

---

## 🎯 Mục tiêu & Câu hỏi nghiên cứu

### Mục tiêu

- Áp dụng quy trình Machine Learning:

  **Tiền xử lý → EDA → Feature Analysis → Modeling → Evaluation → Model Comparison**

- Phân tích mối quan hệ giữa các đặc trưng và mức độ béo phì
- Xác định các đặc trưng quan trọng trong việc phân loại mức độ béo phì
- Thực nghiệm và so sánh 5 mô hình Machine Learning
- Áp dụng Hydra để quản lý configuration
- Xác định mô hình có hiệu năng tốt nhất dựa trên các metrics đánh giá

### Câu hỏi nghiên cứu

1. Những đặc trưng nào có mối quan hệ với mức độ béo phì?
2. Những đặc trưng nào đóng vai trò quan trọng trong việc phân loại mức độ béo phì?
3. Có thể sử dụng Machine Learning để phân loại mức độ béo phì của một cá nhân không?
4. Mô hình nào trong 5 mô hình được thử nghiệm có hiệu năng tốt nhất?
5. Việc loại bỏ các đặc trưng ít quan trọng có ảnh hưởng như thế nào đến hiệu năng mô hình?

---

## 📂 Dataset

- **Tên:** Obesity Levels
- **Nguồn:** Public Dataset ([Kaggle – Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels))
- **Số đặc trưng:** 17 features
- **Loại bài toán:** Multi-class Classification
- **Đối tượng:** Dữ liệu cá nhân liên quan đến mức độ béo phì
- **Phạm vi dữ liệu:** Mexico, Peru và Colombia

### Một số thuộc tính quan trọng

- `age`: Tuổi
- `gender`: Giới tính
- `height`: Chiều cao
- `weight`: Cân nặng
- `family_history_with_overweight`: Tiền sử gia đình có người thừa cân
- `FAVC`: Thường xuyên sử dụng thực phẩm giàu năng lượng
- `FCVC`: Tần suất sử dụng rau
- `NCP`: Số lượng bữa ăn chính
- `CAEC`: Tần suất ăn giữa các bữa
- `SMOKE`: Tình trạng hút thuốc
- `CH2O`: Lượng nước uống
- `SCC`: Theo dõi lượng calories
- `FAF`: Tần suất hoạt động thể chất
- `TUE`: Thời gian sử dụng thiết bị công nghệ
- `CALC`: Tần suất sử dụng đồ uống có cồn
- `MTRANS`: Phương tiện di chuyển
- `obesity_level`: Mức độ béo phì – **Target**

---

## 🧪 Quy trình Machine Learning

### 1️⃣ Tiền xử lý dữ liệu

- Đọc và kiểm tra dữ liệu
- Kiểm tra kiểu dữ liệu
- Chuẩn hóa tên các cột
- Kiểm tra missing values
- Kiểm tra dữ liệu trùng lặp
- Xử lý các biến categorical

---

### 2️⃣ Phân tích khám phá dữ liệu (EDA)

Thực hiện Exploratory Data Analysis nhằm hiểu rõ đặc điểm của dataset:

- Thống kê mô tả dữ liệu
- Histogram
- Boxplot
- Countplot
- Scatter plot
- Heatmap tương quan
- Phân tích phân phối của các lớp `obesity_level`
- Phân tích mối quan hệ giữa các đặc trưng

Một số phân tích được thực hiện:

- Phân tích mối quan hệ giữa `age` và `obesity_level`
- Phân tích mối quan hệ giữa `weight` và `obesity_level`
- Phân tích mối quan hệ giữa `height` và `weight`
- Phân tích hoạt động thể chất và mức độ béo phì
- Phân tích thời gian sử dụng thiết bị công nghệ
- Phân tích các yếu tố liên quan đến thói quen ăn uống

---

### 3️⃣ Feature Analysis

Thực hiện phân tích mức độ quan trọng của các đặc trưng bằng phương pháp **Chi-Square**.

Top 10 đặc trưng theo Chi-Square Score:

| Rank | Feature | Chi-Square Score |
|---|---|---:|
| 1 | `weight` | 436.59 |
| 2 | `gender` | 326.05 |
| 3 | `monitors_calories_daily` | 122.14 |
| 4 | `food_between_meal` | 113.21 |
| 5 | `number_of_main_meal` | 110.43 |
| 6 | `family_history_with_overweight` | 108.03 |
| 7 | `time_using_technology` | 105.37 |
| 8 | `age` | 91.84 |
| 9 | `transportation` | 42.82 |
| 10 | `freq_of_physical_activity` | 39.59 |

---

## 🤖 Các mô hình Machine Learning được sử dụng

Project thực hiện huấn luyện và so sánh 5 thuật toán:

### 🔹 1. Logistic Regression

Được sử dụng làm mô hình baseline để đánh giá performance cơ bản của bài toán phân loại.

### 🔹 2. Decision Tree

Mô hình cây quyết định có khả năng biểu diễn các mối quan hệ phi tuyến và dễ dàng diễn giải.

### 🔹 3. Random Forest

Mô hình Ensemble kết hợp nhiều Decision Tree nhằm cải thiện khả năng tổng quát hóa.

### 🔹 4. Support Vector Machine

Mô hình phân loại dựa trên việc tìm hyperplane tối ưu để phân tách các lớp dữ liệu.

### 🔹 5. K-Nearest Neighbors

Mô hình phân loại dựa trên khoảng cách giữa một sample và các điểm dữ liệu lân cận.

---

## ⚙️ Configuration Management với Hydra

Project sử dụng **Hydra** để quản lý configuration và các tham số của quá trình training.

Hydra giúp:

- Quản lý configuration tập trung
- Tách code và configuration
- Dễ dàng thay đổi tham số mô hình
- Hỗ trợ chạy nhiều experiment
- Tăng khả năng tái lập experiment

---

## 📊 Kết quả & Đánh giá mô hình

Các mô hình được đánh giá trên tập test gồm **418 samples**.

Các metrics được sử dụng:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

#### 🏆 Model Performance

| Model                  | Accuracy | Macro Precision | Macro Recall | Macro F1-score | Weighted F1-score |
| ---------------------- | -------: | --------------: | -----------: | -------------: | ----------------: |
| Logistic Regression    |     0.76 |            0.76 |         0.75 |           0.75 |              0.76 |
| Decision Tree          | **0.94** |            0.94 |     **0.95** |       **0.94** |          **0.95** |
| Random Forest          | **0.94** |        **0.95** |         0.94 |       **0.94** |              0.94 |
| Support Vector Machine |     0.85 |            0.89 |         0.85 |           0.86 |              0.86 |
| K-Nearest Neighbors    |     0.87 |            0.90 |         0.87 |           0.88 |              0.88 |

### 🏅 Phân tích kết quả
#### 🥇 Decision Tree

Decision Tree đạt:
- Accuracy: **94%**
- Macro F1-score: **0.94**
- Weighted F1-score: **0.95**

Đây là mô hình có **Weighted F1-score** cao nhất trong 5 mô hình.

Mô hình thể hiện performance tốt trên hầu hết các lớp và đặc biệt có kết quả tốt ở các lớp:
- Class 3: F1 = **0.98**
- Class 4: F1 = **0.99**
- Class 5: F1 = **0.97**
- Class 6: F1 = **0.93**

#### 🥈 Random Forest

Random Forest đạt:
- Accuracy: **94%**
- Macro Precision: **0.95**
- Macro F1-score: **0.94**
- Weighted F1-score: **0.94**

Random Forest có khả năng phân loại tốt trên nhiều lớp.

Đặc biệt:
- Class 4: F1 = **1.00**
- Class 2: F1 = **0.96**
- Class 3: F1 = **0.98**
Random Forest cũng đạt **Macro Precision** cao nhất (**0.95**).


Điều này cho thấy mối quan hệ giữa các đặc trưng và biến mục tiêu có thể không hoàn toàn tuyến tính,
do đó các mô hình có khả năng học quan hệ phi tuyến như **Decision Tree** và **Random Forest**
có performance tốt hơn.

### 📌 So sánh tổng quan

Dựa trên kết quả thực nghiệm:

- 🥇 Decision Tree đạt Accuracy **94%** và Weighted F1-score **0.95**
- 🥈 Random Forest đạt Accuracy **94%** và Macro Precision **0.95**
- 🥉 KNN đạt Accuracy **87%**

#### 🏆 Final Recommendation

Dựa trên **Accuracy, Macro F1-score** và **Weighted F1-score**, có thể ưu tiên:

> **Decision Tree** là mô hình có performance tổng thể tốt nhất trong experiment hiện tại.

Tuy nhiên, **Random Forest** cũng là một lựa chọn cạnh tranh với Accuracy tương đương và
Macro Precision cao hơn.

Việc lựa chọn mô hình cuối cùng nên phụ thuộc vào mục tiêu triển khai thực tế,
đặc biệt là metric ưu tiên và khả năng giải thích mô hình.

---

## 🧠 Key Findings & Insights
### 🔍 1. Các đặc trưng quan trọng

Kết quả phân tích cho thấy các đặc trưng nổi bật bao gồm:
- `weight`
- `height`
- `gender`
- `age`
- `food_between_meal`
- `freq_of_physical_activity`
- `family_history_with_overweight`
- `time_using_technology`

Trong đó, `weight` và `gender` có Chi-Square Score cao nhất trong phân tích feature selection.

### 🔍 2. Mối quan hệ giữa tuổi và mức độ béo phì

Dữ liệu cho thấy các mức độ thừa cân và béo phì xuất hiện nhiều ở nhóm tuổi khoảng:

> **15 – 35 tuổi**

Trong đó, một số nhóm mức độ béo phì có xu hướng tập trung ở khoảng tuổi từ 20 – 30.

### 🔍 3. Mối quan hệ giữa chiều cao và cân nặng

`height` và `weight` có xu hướng tương quan dương:

> Chiều cao tăng thường đi kèm với xu hướng cân nặng tăng.

Tuy nhiên, mối quan hệ này không đồng nghĩa với việc chiều cao trực tiếp gây ra tình trạng béo phì.

### 🔍 4. Hoạt động thể chất

Kết quả EDA cho thấy có sự khác biệt về tần suất hoạt động thể chất giữa các nhóm
mức độ béo phì.

Nhìn chung, các nhóm có mức độ béo phì cao hơn có xu hướng hoạt động thể chất thấp hơn.

### 🔍 5. Thời gian sử dụng thiết bị công nghệ

Thời gian sử dụng thiết bị công nghệ có sự khác biệt giữa các nhóm mức độ béo phì.

Tuy nhiên, cần thận trọng khi diễn giải mối quan hệ này vì tương quan không đồng nghĩa với quan hệ nhân quả.

### 📊 Phân tích tương quan

| Feature | Correlation | Relationship | Interpretation |
|---|---:|---|---|
| `weight` | **+0.39** | 🟢 Positive | Tương quan dương yếu với `obesity_level` |
| `food_between_meal` | **+0.34** | 🟢 Positive | Tương quan dương yếu với `obesity_level` |
| `family_history_with_overweight` | **+0.33** | 🟢 Positive | Tương quan dương yếu với `obesity_level` |
| `age` | **+0.23** | 🟢 Positive | Tương quan dương rất yếu với `obesity_level` |
| `freq_of_physical_activity` | **-0.13** | 🔵 Negative | Tương quan âm rất yếu với `obesity_level` |
| `smoking` | **-0.13** | 🔵 Negative | Tương quan âm rất yếu với `obesity_level` |
| `gender` | **+0.01** | ⚪ Very Weak | Gần như không có tương quan tuyến tính với `obesity_level` |

 > ⚠️ Các hệ số tương quan trên chỉ phản ánh mối quan hệ tuyến tính giữa các biến. Không nên diễn giải trực tiếp thành quan hệ nhân quả.

### 🔎 Phân tích dữ liệu ngoại lai

Thông qua Boxplot, một số biến có thể xuất hiện các giá trị ngoại lai: `number_of_main_meal` - `age`

Các giá trị này cần được kiểm tra thêm trước khi quyết định loại bỏ.

--- 

## 🗂️ Cấu trúc thư mục

```
deploy_obesity_classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── confs/
│   ├── __init__.py
│   ├── config.yaml
│   ├── features.yaml
│   ├── models.yaml
│   └── preprocessing.yaml
│
├── data/
│   └── data_obesity.csv
│
└── src/
   ├── __init__.py
   │
   ├── common/
   │   ├── __init__.py
   │   ├── load_data_preprocess.py
   │   ├── utils.py
   │   ├── utils_logging.py
   │   └── visualization.py
   │
   ├── evaluation/
   │   ├── __init__.py│
   │   ├── eval_model.py
   │   └── importance.py
   │
   └── train/
        ├── __init__.py
        ├── preprocessing_to_train.py
        └── train_model.py
```

---

## 🚀 Cài đặt

### Clone repository

```bash
git clone deploy_obesity_classification
cd deploy_obesity_classification
```

### Tạo Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

--- 

## ▶️ Training

### Train model mặc định

```bash
python main.py
```

### Train 5 model riêng

```bash
python main.py model=<tên model train>
```

---

## ⚠️ Hạn chế
- Dataset phục vụ mục đích nghiên cứu và học tập, không phải dữ liệu lâm sàng.
- Kết quả mô hình không thể thay thế chẩn đoán y tế.
- Chưa thực hiện Hyperparameter Optimization chuyên sâu.
- Chưa triển khai Model Registry hoàn chỉnh.
- Chưa đánh giá model bằng Cross-Validation chuyên sâu.
- Chưa triển khai hệ thống Model Serving.
- Chưa đánh giá Model Drift và Data Drift.

## 🚀 Hướng phát triển
- Cross-Validation, Feature Selection, Feature Engineering
- Hoàn thiện MLflow Model Registry
- CI/CD Pipeline
- Docker
- Xây dựng giao diện dự đoán với Streamlit
- Explainable AI với SHAP

 --- 
 
## 👨‍🎓 Thông tin học thuật
- **Loại project:** Academic Project
- **Chủ đề:** Machine Learning / Data Science
- **Bài toán:** Multi-class Classification
- **Configuration Management:** Hydra

> Project được thực hiện nhằm mục đích học tập và nghiên cứu về quy trình xây dựng
Machine Learning Pipeline, từ bước tiền xử lý dữ liệu, phân tích dữ liệu,
feature analysis, huấn luyện mô hình đến đánh giá và so sánh mô hình.
