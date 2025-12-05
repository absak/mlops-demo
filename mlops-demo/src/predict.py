import sys
import json
import requests

url = "http://localhost:5000/invocations"
headers = {"Content-Type": "application/json"}

def predict_from_args():
    # Exemple : python predict.py 5.1 3.5 1.4 0.2
    values = [float(x) for x in sys.argv[1:]]
    data = {"inputs": [values]}
    return data

def predict_from_file():
    # Exemple : python predict.py input.json
    filename = sys.argv[1]
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def predict_interactive():
    # Exemple : python predict.py (puis saisir les valeurs)
    values = input("Entrez les 4 caractéristiques séparées par des espaces : ")
    values = [float(x) for x in values.split()]
    data = {"inputs": [values]}
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(".json"):
            data = predict_from_file()
        else:
            data = predict_from_args()
    else:
        data = predict_interactive()

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("Prédiction :", response.json())
