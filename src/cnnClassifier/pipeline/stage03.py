from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_training_component import Training
from src.cnnClassifier.utils.common import *
from src.cnnClassifier import logger



class TrainingPipeline:

    def __init__(self):
        pass


    def main(self):
        try:
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()
        except Exception as e:
            raise e
