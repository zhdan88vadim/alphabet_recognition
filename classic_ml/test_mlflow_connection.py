import mlflow

# Устанавливаем URI
mlflow.set_tracking_uri("sqlite:////media/vadim/1TB_SSD/my_github/alphabet_recognition/deep_learning/mlflow.db")

# Проверяем
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Пробуем создать эксперимент
experiment_name = "test_connection"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_param("test", "connection_works")
    print("✅ MLflow работает!")