# mlops-demo/data/generate_iris.py
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv("iris.csv", index=False)
print("iris.csv créé")
