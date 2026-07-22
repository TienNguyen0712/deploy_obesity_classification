import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


for name, model in models.items():
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
  plt.title(f'Confusion Matrix - {name}')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()

# Xuất ra điểm của các mô hình dự đoán
for name, model in models.items():
  train_score = model.score(x_train, y_train) * 100
  test_score = model.score(x_test, y_test) * 100
  print(f"{name}: \n Train Score: {train_score:.2f} %\n Test Score: {test_score:.2f} %") # Xuất điểm kết quả của các mô hình ra màn hình
  print("-" * 30)
