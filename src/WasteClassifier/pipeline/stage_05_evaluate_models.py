"""Unified Evaluation Stage: Evaluate DL + ML models

This stage evaluates all runs that have persisted MLflow run context files under
`tracking.runs_root_dir` and generates:
- confusion matrices
- DL vs ML comparison plots
- evaluation metrics json

Author: Grzegorz Gomza
Date: March 2026
"""

from WasteClassifier.config.configuration import ConfigurationManager
from WasteClassifier.components.evaluation.evaluate_models import EvaluateModels
from WasteClassifier import logger


class UnifiedEvaluatePipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            tracking_config = config.get_tracking_config()
            visualization_config = config.get_visualization_config()

            evaluator = EvaluateModels(
                evaluation_config=evaluation_config,
                tracking_config=tracking_config,
                visualization_config=visualization_config,
            )
            evaluator.main()

            logger.info("Unified evaluation pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in unified evaluation pipeline: {e}")
            raise e


if __name__ == "__main__":
    pipeline = UnifiedEvaluatePipeline()
    pipeline.main()
