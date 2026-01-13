from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_evaluate import Evaluation
from src.cnnClassifier import logger
import os


STAGE_NAME="Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Viveksasane/Deep_Learning.mlflow" # Dagshub lin
        os.environ["MLFLOW_TRACKING_USERNAME"]="Viveksasane" # github link
        os.environ["MLFLOW_TRACKING_PASSWORD"]="520d21c2226e27a8ddcf88e5e150a5cff3894c66"

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()