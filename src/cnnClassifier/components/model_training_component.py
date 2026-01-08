import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier.entity.config import TrainingConfig




class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


   
    def get_base_model(self):
        self.model = tf.keras.models.load_model(   # For loading the model
            self.config.updated_base_model_path
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',  # Or your actual loss function
                           metrics=['accuracy']
)


    def train_valid_generator(self):


        datagenerator_kwargs = dict(
            rescale = 1./255, # 1. is a float value
            validation_split=0.20
        )


        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # It will remove 3 from the params.yaml
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


        if self.config.params_is_augmentation:   # It will check whether the params file augmetation is True if than it will excute if block 
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator


        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )


   
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)






   
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size   # 60000 / 16 = 6000 it will this much only images
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size


        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )


        self.save_model(  # Here path for saving the model
            path=self.config.trained_model_path,
            model=self.model
        )
