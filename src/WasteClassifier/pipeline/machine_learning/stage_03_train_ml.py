"""ML Pipeline Stage: Train Machine Learning Model (XGBoost)

Author: Grzegorz Gomza
Date: March 2026
"""

from WasteClassifier.config.configuration import ConfigurationManager
from WasteClassifier.components.machine_learning.train_xgb import TrainXGBModel
from WasteClassifier import logger


class MLTrainModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            train_ml_config = config.get_train_ml_model_config()
            tracking_config = config.get_tracking_config()
            visualization_config = config.get_visualization_config()

            trainer = TrainXGBModel(
                config=train_ml_config,
                tracking_config=tracking_config,
                visualization_config=visualization_config,
            )
            trainer.main()

            logger.info("ML training pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in ML training pipeline: {e}")
            raise e


if __name__ == "__main__":
    pipeline = MLTrainModelPipeline()
    pipeline.main()
