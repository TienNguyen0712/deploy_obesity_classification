# Lựa chọn các đặc trưng bằng 3 phương pháp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Tối ưu mô hình
from sklearn.ensemble import GradientBoostingClassifier

# Khai báo mô hình
from sklearn.linear_model import LogisticRegression # Hồi qui tuyến tính
from sklearn.tree import DecisionTreeClassifier # Cây quyết định
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.svm import SVC # Hỗ trợ SVM
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors


def importance(x_train, y_train, model, select):
    if hasattr(model, 'feature_importances_'):
        importance_model = model
    else:
        print('Sử dụng GradientBoosting để tìm đặc trưng quan trọng')
        importance_model = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
        importance_model.fit(x_train, y_train)

    # Đặc trưng quan trọng
    feature_importances = pd.Series(importance_model.feature_importances_, index=select)
    feature_importances.sort_values(ascending=True, inplace=True)

    # Plot
    feature_importances.plot.bar(figsize=(20, 6))
    plt.title(f'Feature Importances With {model}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

# Thực hiện tìm kiếm đặc trưng quan trọng
select = ['age', 'gender', 'height', 'weight', 'freq_of_alcohol',
       'freq_high_calories_food', 'freq_of_vegetable', 'number_of_main_meal',
       'monitors_calories_daily', 'smoking', 'daily_water',
       'family_history_with_overweight', 'freq_of_physical_activity',
       'time_using_technology', 'food_between_meal', 'transportation']

target = ['obesity_level']

x_new = df[select]
y_new = df[target]

model = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for name, model in models.items():
  model.fit(x_new, y_new)
  y_pred = model.predict(x_new)
  print(f"{name}: \n")
  importance(x_new, y_new, model, select)
  print("-" * 50)


x_importance = x.drop('index', axis = 1)
y_importance = y

best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(abs(x_importance), y_importance)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x_importance.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
feature_importance = featureScores.nlargest(10, 'Score').sort_values(by='Score', ascending = False)

print("Top 10 đặc trưng quan trọng theo phương pháp chi2: \n", feature_importance)
