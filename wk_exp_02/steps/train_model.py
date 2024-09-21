from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import joblib
import mlflow
from typing import Tuple, Dict, Union, Annotated
from zenml import step, Model, get_step_context, log_model_metadata
from datetime import date
import logging
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

# Get today's date
today = date.today()
## Model Directory
model_dir = 'model_dir'


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    data: pd.DataFrame,
    model_name: Union[str, None]
) -> Annotated[Model, "trained_model"]:
    #Tuple[Annotated[Model, "trained_model"], Annotated[Dict[str, float], "RMSE"]]:
    y = data['Salary']
    X = data[['Experience Years']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.7,
                                                        random_state=234)

    model = None
    if model_name == "linearRegression" or model_name is None:
        model = LinearRegression()
    elif model_name == "decisionTree":
        model = DecisionTreeRegressor()
    elif model_name == "randomForest":
        model = RandomForestRegressor()

    # Log the model type
    mlflow.log_param("model_type", model_name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log Metrics
    mlflow.log_metric("rmse", rmse)

    # Log the sklearn model
    mlflow.sklearn.log_model(model, "salary_prediction_model")

    # Create a ZenML Model object
    zenml_model = Model(
        name="salary_prediction_model",
        model=model,
        metadata={"rmse": str(rmse)}
    )

    # Log metadata directly to the zenml_model object
    zenml_model.log_metadata({"rmse": str(rmse)})
    zenml_model.set_stage("staging", force=True)
    # Export Model to Local Directory
    model_dir_name = f"{rmse}_{model_name}_{today}.pkl"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_dir_name)

    # Dump model
    joblib.dump(model, model_path)

    # Log the model path as an artifact
    mlflow.log_artifact(model_path)

    logging.info(f"Successfully Trained {model_name} \n"
                 f"and stored trained model to Model Register: {model_path}")

    return zenml_model #, {"rmse": rmse}
