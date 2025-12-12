import boto3
import json

# Nom de ton endpoint SageMaker (doit correspondre à celui défini dans deploy_endpoint.py)
ENDPOINT_NAME = "iris-rf-endpoint"

def test_inference():
    # Client SageMaker Runtime
    runtime = boto3.client("sagemaker-runtime")

    # Exemple de données Iris (Setosa)
    payload = {
        "instances": [[5.1, 3.5, 1.4, 0.2]]
    }

    # Appel du endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    # Lecture de la réponse
    result = json.loads(response["Body"].read())
    print("✅ Réponse du modèle :", result)

if __name__ == "__main__":
    test_inference()

