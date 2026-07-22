from typing import Dict, Any

import pandas as pd 

# Khai báo mô hình
from sklearn.linear_model import LogisticRegression # Hồi qui tuyến tính
from sklearn.tree import DecisionTreeClassifier # Cây quyết định
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.svm import SVC # Hỗ trợ SVM
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors

# Đánh giá mô hình
from sklearn.metrics import classification_report

from omegaconf import DictConfig
from src.common.utils_logging import logger

MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
}

def train_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, cfg: DictConfig) -> Dict[str, Any]:
  """
  Huấn luyện nhiều mô hình theo cáu hình hydra
  """
  results = {}
  for model_key, model_cfg in cfg.models.items():
    model_name = model_cfg.name
    model_class_name = model_cfg.class_name

    # Lấy class model 
    model_class = MODEL_CLASSES[model_class_name]

    # Chuyền config param thành dict
    params = dict(model_cfg.params)

    # Khởi tạo model
    model = model_class(**params)
    print("-" * 50)
    logger.info(f" ================= Đang huấn luyện: {model_name} ========= ")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Đánh giá
    logger.info(f" ================= Đang đánh giá: {model_name} ========= ")
    report = classification_report(y_test, y_pred)
    logger.info(report )
    # Lưu kết quả
    results[model_key] = { "model": model, "y_pred": y_pred, "report": report}
  return results