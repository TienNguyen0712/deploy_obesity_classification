from omegaconf import DictConfig

import matplotlib.pyplot as plt
import seaborn as sns



def get_info(df):
    return df.info()

def get_numberical_categorical_features(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    print("Numeric_features: ", numeric_features)
    print("Categorical_features: ", categorical_features)

def get_describe(df, include="numberic"):
    return df.describe(include)


def check_null(df):
    missing_data = df.isnull().sum()

    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Data')
    plt.show()

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
