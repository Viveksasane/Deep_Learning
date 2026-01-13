import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.cnnClassifier.entity.config import EvaluationConfig
from src.cnnClassifier.utils.common import *

### It is used for checking 


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


   
    def _valid_generator(self):


        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )


        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )


        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )


        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )




    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
   


    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()


    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)



    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # ✅ Model path
        model_path = Path("artifacts/training/model.keras")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with mlflow.start_run() as run:
            # Log params & metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
            "loss": self.score[0],
            "accuracy": self.score[1]
        })

        # ✅ Save model
        self.model.save(model_path)

        # ✅ Log model as artifact
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # ✅ Register model only if not local file store
        if tracking_url_type_store != "file":
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="VGG16Model"
            )
