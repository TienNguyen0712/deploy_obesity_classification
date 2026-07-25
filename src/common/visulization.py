from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.utils import save_figure

# Diều chỉnh màu
COLORS = ['#e9e9ea', '#9c94b4', '#6dbc6e', '#3c9c7c', '#b2a4c0', '#7f8e9a', '#94b4bc']

def plot_hist_of_numberical_col(df: pd.DataFrame , cfg: DictConfig):
# Biểu đồ với cột dữ liệu số (Numerical)
    for col in list(cfg.numerical_features):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5)) # 2 hàng và 1 cột

        # Hiện biểu đồ phân tán (Biểu đồ trên)
        sns.histplot(data = df, x = col, kde = True, ax = axes[0], color='darkblue')
        axes[0].set_title(f'Phân bố của cột {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Tần suất')

        # Hiện biểu đồ boxplot (Biểu đồ dưới)
        sns.boxplot(data = df, x = col, ax = axes[1], color = '#44a47c')
        axes[1].set_title(f'Boxplot của cột {col}')
        axes[1].set_xlabel(col)

        save_figure(fig, cfg, "numerical_col", f"{col}-col.png")


def plot_hist_of_categorical_col(df: pd.DataFrame, cfg: DictConfig):
# Biểu đồ cột dữ liệu Phân loại (Categorical)
    for col in list(cfg.categorical_features):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Đếm giá trị cho từng thuộc tính
        value_counts = df[col].value_counts()

        # Tạo biểu đồ bar plot
        fig = sns.barplot(x = value_counts.values, y = value_counts.index, palette='viridis', ax=ax)

        # Điều chỉnh biểu đồ tên cột y và tên cột x
        ax.set_title(f"Phân bố của cột {col}")
        ax.set_ylabel(col)
        ax.set_xlabel("Số lượng")
        ax.set_xticks(rotation=45)
        ax.set_grid(axis = 'x', alpha = 0.75)
        save_figure(fig, cfg, "categorical_col", f"{col}-col.png")


    # Thêm số lượng
        for index, value in enumerate(value_counts.values):
            plt.text(value, index, f'{value}', va = 'center')




def plot_pie_of_categorical_features(df: pd.DataFrame, cfg: DictConfig):
    for col in list(cfg.categorical_features):
        plt.figure(figsize=(10, 5))

        # Đếm giá trị cho từng thuộc tính
        value_counts = df[col].value_counts()

        # Vẽ biểu đồ tròn
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=COLORS)
        plt.title(f"Phân bố của cột {col}")
        plt.axis('equal')

        # Điều chỉnh hiện thị của biểu đổ
        plt.legend()


def plot_subplot_of_target_to_others(df: pd.DataFrame, cfg: DictConfig):
    target = df[cfg.target_column]

    # Danh sách các cột cần group by 'obesity_level' và đếm giá trị
    columns_to_group = list(cfg.columns_to_group)
    # Tạo dictionary lưu trữ kết quả groupby
    grouped_data = {col: df.groupby(cfg.target_column)[col].value_counts() for col in columns_to_group}

    # Tạo figure và axes
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()  # Chuyển mảng 2D thành 1D để dễ duyệt

    # Màu sắc cho các biểu đồ (tùy chỉnh)
    colors = sns.color_palette('viridis', n_colors=len(df[cfg.target_column].unique()))

    # Lặp qua các cột và vẽ biểu đồ
    for i, (col_name, grouped_series) in enumerate(grouped_data.items()):
        if i < len(axes):
            grouped_series.unstack().plot(kind='bar', ax=axes[i], color=colors)
            axes[i].set_title(f'Obesity Level vs {col_name.replace("_", " ").title()}', fontsize=14)
            axes[i].set_xlabel('Obesity Level', fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            axes[i].legend(title=col_name.replace("_", " ").title())

    # Loại bỏ các subplot trống nếu số lượng cột ít hơn 9
    if len(columns_to_group) < 9:
        for j in range(len(columns_to_group), 9):
            fig.delaxes(axes[j])


def plot_kde_plot(df: pd.DataFrame, cfg: DictConfig):
    for col in cfg.numeric_features:
        plt.figure(figsize=(10, 5))

        # Biểu đồ bar hiển thị so sánh giữa các biến phân loại với mức độ béo phì
        sns.kdeplot(data = df, x = col, hue = cfg.target_column, fill = True, palette='viridis')
        plt.title(f"Biểu đồ KDE biểu thị phân tán của cột {col} so với mức độ béo phì")

        # Điều chỉnh biểu đồ tên cột y và tên cột x
        plt.ylabel(col)
        plt.xlabel("Mức độ béo phì")
        plt.xticks(rotation=45)


def plot_pairplot(df: pd.DataFrame, cfg: DictConfig):
    sns.pairplot(data = df.drop(cfg.drop_columns, axis = 1), hue = 'gender', palette='viridis')

def plot_corr(df: pd.DataFrame, cfg: DictConfig):
    df_num = df.drop(cfg.drop_columns, axis=1)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df_num.corr(), annot=True, cmap='viridis', ax=ax)
    ax.set_title('Hệ số tương quan giữa các biến số')
    save_figure(fig, cfg, "numerical_col", "correlation.png")
    
