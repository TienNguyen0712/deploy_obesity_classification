import pandas as pd
from omegaconf import DictConfig
from typing import Optional

from src.common.utils_logging import logger

# Dọc dữ liệu và trả về dữ liệu
def read(cfg: DictConfig) -> Optional[pd.DataFrame]:
    data_path = cfg.paths.data_path
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột")
        return df
    except Exception as e:
        logger.error("Lỗi khi tải dữ liệu obesity!", exc_info=True)
        return None

# Đổi tên các cột đặc trưng
def rename_columns(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    df = df.rename(columns=dict(cfg.rename_columns)) # Đổi tên cột

    # Dổi các giá trị trong transportation, obesity_level, gender, freq_high_calories_food
    for col, mapping in cfg.value_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(dict(mapping))
    logger.info(f"Đã đổi tên các đặc trưng")

    return df

def drop_dup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index() # Xóa dữ liệu bị trùng
    logger.info("Đã loại bỏ dữ liệu trùng")
    return df
