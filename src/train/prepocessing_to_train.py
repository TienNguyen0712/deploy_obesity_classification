from omegaconf import DictConfig
import pandas as pd

# Tiền xử lý dữ liệu
from sklearn.preprocessing import OneHotEncoder # Biến phân loại
from sklearn.preprocessing import StandardScaler # Biên số
from sklearn.model_selection import train_test_split

from src.common.utils_logging import logger

def prepocessing(x_train: pd.DataFrame, x_test: pd.DataFrame, cfg: DictConfig):
  """
  Fit Encoder và Scaler trên X_train,
  sau đó transform X_train và X_test.

  Returns:
      x_train_processed
      x_test_processed
      preprocessor
  """

  categorical_features = list(cfg.categorical_features)
  numeric_features = list(cfg.numerical_features)

  # Phân loại
  encoder = OneHotEncoder(handle_unknown = cfg.encoding.handle_unknown, sparse_output=False)
  encoder.fit(x_train[categorical_features])

  x_train_encoded = encoder.transform(x_train[categorical_features])
  x_test_encoded = encoder.transform(x_test[categorical_features])

  # Số
  scaler = StandardScaler()
  scaler.fit(x_train[numeric_features])

  x_train_scaled = scaler.transform(x_train[numeric_features])
  x_test_scaled = scaler.transform(x_test[numeric_features])

  #Lấy tên cột sau OneHotEncoder
  encoded_columns = ( encoder.get_feature_names_out(categorical_features))

  # Tạo DataFrame cho dữ liệu đã encode
  x_train_encoded = pd.DataFrame( x_train_encoded, columns=encoded_columns, index=x_train.index)
  x_test_encoded = pd.DataFrame(x_test_encoded, columns=encoded_columns, index=x_test.index)

  # Tạo DataFrame cho dữ liệu đã scale
  x_train_scaled = pd.DataFrame( x_train_scaled, columns=numeric_features, index=x_train.index)
  x_test_scaled = pd.DataFrame( x_test_scaled, columns=numeric_features, index=x_test.index)

  # Kết hợp Numerical + Categorical
  x_train_processed = pd.concat([ x_train_scaled, x_train_encoded], axis=1)
  x_test_processed = pd.concat([x_test_scaled, x_test_encoded], axis=1)

  # Trả về kết quả
  return ( x_train_processed, x_test_processed, encoder, scaler )

def split_data(df: pd.DataFrame, cfg: DictConfig):     
  x = df.drop(cfg.target_column, axis=1) # Lựa chọn biến có thể dự đoán
  y = df[cfg.target_column] # Biến mục tiêu
  # Sử dụng stratify nếu được bật trong config
  stratify = (
    y if cfg.stratify else None
  )

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify)
  logger.info("Đã chia thành x_train, y_train, x_test, y_test")
  return x_train, x_test, y_train, y_test

