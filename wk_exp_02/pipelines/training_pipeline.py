from zenml import pipeline
from steps.ingest_data import load_data
from steps.train_model import train_model
from steps.promote_model import promote_model
# from zenml.integrations.mlflow.steps.mlflow_registry import (
#     mlflow_register_model_step,
# )

@pipeline
def simple_ml_pipeline(modelname: str):
    dataset = load_data()
    model = train_model(dataset, modelname)
    is_promoted = promote_model(model)

    return is_promoted

