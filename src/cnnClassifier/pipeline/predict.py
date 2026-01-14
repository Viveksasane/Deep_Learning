import numpy as np
import tensorflow
from keras.models import load_model
from keras.preprocessing import image
import os






class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename




   
    def predict(self):
        # load model
        model = load_model(os.path.join("model","model.keras"))


        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224)) # It i
        test_image = image.img_to_array(test_image) # Converting to array
        test_image = np.expand_dims(test_image, axis = 0)  # It will give answer as 1 , 224 , 224
        result = np.argmax(model.predict(test_image), axis=1) # It will predict the result
        print(result)


        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]
