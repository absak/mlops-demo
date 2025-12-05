# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib, os
import numpy as np

# Config MLflow (chemin absolu pour éviter les confusions)
mlflow.set_tracking_uri("file:///C:/Users/abdel/demoDevOps/mlops-demo/mlops-demo/mlruns")
mlflow.set_experiment("mlops-demo")

# Charger dataset
df = pd.read_csv("file:///C:/Users/abdel/demoDevOps/mlops-demo/mlops-demo/data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")

    # Sauvegarde locale
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/model.joblib")

    # Tracking MLflow
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_artifact("../models/model.joblib")

    # Exemple d’entrée pour signature MLflow
    input_example = np.array([X_train.iloc[0].tolist()])
    mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

    # ✅ Affichage du Run ID
    print("Run ID :", run.info.run_id)

print("✅ Modèle entraîné, sauvegardé et loggé dans MLflow")
