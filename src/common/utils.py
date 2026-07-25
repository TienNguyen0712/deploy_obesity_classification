from omegaconf import DictConfig
from pathlib import Path 

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.common.utils_logging import logger

def get_info(df):
    print("Thông tin của bảng")
    print(df.info())

def get_numberical_categorical_features(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    print("Numeric_features: ", numeric_features)
    print("Categorical_features: ", categorical_features)

def get_describe(df, include="numberic"):
    print(df.describe(include))

def save_figure(fig, cfg: DictConfig, folder_name: str, file_name: str):
    """
    Lưu fig vào thưu mục riêng fig: Là fig cần lưu
    """
    output_dir = Path(cfg.output.fig_dir)

    # Tạo thư mục riêng 
    save_dir = output_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Đường dẫn đầy dủ
    file_path = save_dir / file_name

    # Lưu hình 
    fig.savefig(
        file_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)
    logger.info(f"Đã lưu hình tại: {file_path}")


def check_null(df: pd.DataFrame, cfg: DictConfig):
    missing_data = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Data')

    save_figure(fig, cfg, "missing_data", "missing_data.png")

    print("Dữ liệu bị thiếu: \n", missing_data)
    print("-" * 50)

    for col in df.columns:
        if df[col].isnull().any():
            print(f"{col} có dữ liệu bị thiếu")
        else:
            print("Không dữ liệu bị thiếu")


def check_duplicated(df):
    print("Dữ liệu bị trùng có: ", df.duplicated().sum(), " dòng")


def get_unique_of_categorical(df):
    print("Các thuộc tính khác nhau trong cột phân loại: \n")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].unique()}\n")

def get_correlation_with_target(df, cfg: DictConfig): 
    return df.corr()[cfg.target_column].sort_values(ascending = False)
