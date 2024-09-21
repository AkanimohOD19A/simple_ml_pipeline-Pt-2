from zenml import step, Model, log_model_metadata
from typing import Dict, Annotated
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)


@step
def promote_model(
        model: Model,
        # metrics: Dict[str, float],
        stage: str = "production"
) -> Annotated[bool, "best_model"]:
    # Get the ZenML client
    client = Client()

    # Staging RMSE
    current_staging = client.get_model_version(
        model_name_or_id=model.name,
        model_version_name_or_number_or_id="staging"
    )
    staging_rmse = float(current_staging.run_metadata["rmse"].value)
    logger.critical(f"STAGING RMSE: {staging_rmse}")

    # Production RMSE
    #> Initiate with None & Inf
    current_production = None
    production_rmse = float('inf')

    try:
        # Try to get the production model
        current_production = client.get_model_version(
            model_name_or_id=model.name,
            model_version_name_or_number_or_id="production"
        )
        if current_production:
            production_rmse = float(current_production.run_metadata["rmse"].value)
            logger.info(f"Current production model version: {current_production.id}, RMSE: {production_rmse}")
    except Exception as e:
        logger.warning(f"Error fetching previous production model: {str(e)}")

    # Condition for promotion
    if staging_rmse < production_rmse:
        try:
            # Archive the current production model if it exists
            if current_production:
                try:
                    production_model = Model(
                        name=model.name,
                        version="production"
                    )
                    # This will set this version to production
                    production_model.set_stage(stage="archived", force=True)
                    logger.info(f"Previous production model (id {current_production.id}) archived. \n"
                                f"Previous RMSE: {production_rmse}, New RMSE: {staging_rmse}")
                except Exception as archive_error:
                    logger.warning(f"Failed to archive previous model: {str(archive_error)}")

            log_model_metadata(
                model_name=model.name,
                model_version=model.version,
                metadata={"rmse": staging_rmse},
            )
            # Promote the new model
            model.set_stage(stage, force=True)
            logger.info(f"New model (version {model.version}) promoted to {stage}!")
            return True
        except Exception as e:
            logger.error(f"Error during model promotion: {str(e)}")
            return False
    else:
        logger.info(
            f"Model not promoted. STAGING RMSE ({staging_rmse}) is not better than "
            f"PRODUCTION RMSE ({production_rmse})")
        return False
