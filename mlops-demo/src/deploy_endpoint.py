
import os
from sagemaker.sklearn.model import SKLearnModel

role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("ARTIFACTS_BUCKET")
prefix = "iris/artifacts"
endpoint_name = "iris-rf-endpoint"

model = SKLearnModel(
    model_data=f"s3://{bucket}/{prefix}/model.joblib",
    role=role_arn,
    entry_point="src/predict.py",   # script d’inférence
    framework_version="1.2-1"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name
)

print("✅ Endpoint déployé :", endpoint_name)
