import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

ref = pd.read_csv("data/iris.csv")
cur = pd.read_csv("data/iris.csv")  # à remplacer par prédictions récentes

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("reports/evidently_iris.html")

print("✅ Rapport Evidently généré")

