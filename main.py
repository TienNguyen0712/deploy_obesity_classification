import hydra
from omegaconf import DictConfig

from src.common.utils_logging import logger
from src.common.load_data_prepocess import read, rename_columns, drop_dup
from src.train.train_model  import train_models
from src.train.prepocessing_to_train import prepocessing, split_data

@hydra.main(version_base=None, config_path="confs", config_name="config")

def train(cfg: DictConfig):
    # Đọc dữ liệu
    logger.info("Bắt đầu quá trình huấn luyện, đọc dữ liệu")
    df = read(cfg)  
    # Thực hiện đổi tên 
    rename_df = rename_columns(df, cfg)
    rename_df = drop_dup(rename_df)
    logger.info("Đẫ tiền xử lý dữ liệu")
    # Trực quan hóa 
    # Huấn luyện mô hình 
    (x_train, x_test, y_train, y_test) = split_data(rename_df, cfg)
    # Fit Transform train/test
    (x_train_processed, x_test_processed, encoder, scaler) = prepocessing(x_train, x_test, cfg)
    result = train_models(x_train_processed, x_test_processed, y_train, y_test, cfg)
    logger.info("Đã xong quá trình huấn luyện")

if __name__ == "__main__":
    train()


