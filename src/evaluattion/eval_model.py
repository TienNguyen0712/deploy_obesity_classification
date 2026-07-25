from omegaconf import DictConfig

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.common.utils import save_figure
from src.common.utils_logging import logger

def plot_confusion_matrix(results, y_test: pd.DataFrame, cfg: DictConfig):
  for model_key, result in results.items():
    model_name = cfg.models[model_key].name
    
    y_pred = results[model_key]["y_pred"]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    save_figure(fig, cfg, "confusion_matrix", f"Confusion Matrix - {model_name}.png")

  
# Xuất ra điểm của các mô hình dự đoán
def score_of_model(results, x_train: pd.DataFrame, y_train: pd.DataFrame, 
                     x_test: pd.DataFrame, y_test: pd.DataFrame, cfg: DictConfig):
    for model_key, model_cfg in results.items():
      model_name = cfg.models[model_key].name
      model = results[model_key]["model"]
      train_score = model.score(x_train, y_train) * 100
      test_score = model.score(x_test, y_test) * 100
      logger.info(f"{model_name}: \n Train Score: {train_score:.2f} %\n Test Score: {test_score:.2f} %") # Xuất điểm kết quả của các mô hình ra màn hình
      print("-" * 30)
