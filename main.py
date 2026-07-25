import hydra
from omegaconf import DictConfig

from src.common.utils_logging import logger
from src.common.load_data_prepocess import read, rename_columns, drop_dup
from src.common.visulization import plot_hist_of_numberical_col, plot_corr 
from src.common.utils import check_null

from src.train.train_model  import train_models
from src.train.prepocessing_to_train import prepocessing, split_data

from src.evaluattion.eval_model import plot_confusion_matrix, score_of_model

@hydra.main(version_base=None, config_path="confs", config_name="config")

def train(cfg: DictConfig):
    # Đọc dữ liệu
    logger.info("Bắt đầu quá trình huấn luyện, đọc dữ liệu")
    df = read(cfg)  
    # Thực hiện đổi tên 
    rename_df = rename_columns(df, cfg)
    rename_df = drop_dup(rename_df)
    logger.info("Đã tiền xử lý dữ liệu")
    # Trực quan hóa 
    logger.info("Bắt đầu quá trình trực quan hóa dữ liệu")
    plot_hist_of_numberical_col(rename_df, cfg)
    df_num = rename_df.select_dtypes(exclude="object")
    plot_corr(df_num, cfg)
    # Huấn luyện mô hình 
    (x_train, x_test, y_train, y_test) = split_data(rename_df, cfg)
    # Fit Transform train/test
    (x_train_processed, x_test_processed, encoder, scaler) = prepocessing(x_train, x_test, cfg)
    result = train_models(x_train_processed, x_test_processed, y_train, y_test, cfg)
    logger.info("Đã xong quá trình huấn luyện")
    # Đánh giá mô hình 
    logger.info("Thực hiện đánh giá mô hình huấn luyện")
    plot_confusion_matrix(result, y_test, cfg)
    score_of_model(result, x_train_processed, x_test_processed, y_train, y_test, cfg)
    logger.info("Đã xong quá trình đánh giá")


if __name__ == "__main__":
    train()

