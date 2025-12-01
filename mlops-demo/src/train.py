import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib, os

# Config MLflow
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("mlops-demo")

# Charger dataset
df = pd.read_csv("mlops-demo/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")

    os.makedirs("mlops-demo/models", exist_ok=True)
    joblib.dump(model, "mlops-demo/models/model.joblib")

    mlflow.log_metric("f1_macro", f1)
    mlflow.log_artifact("mlops-demo/models/model.joblib")

print("✅ Modèle entraîné et sauvegardé dans models/model.joblib")
