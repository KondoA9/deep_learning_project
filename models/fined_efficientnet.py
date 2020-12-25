import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB7

class FinedEfficientNet:

    @staticmethod
    def build(input_shape, label_num):
        efficientnet = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
    
        top_model = layers.GlobalAveragePooling2D()(resnet50.output)
        top_model = layers.Dense(4096, activation='relu')(top_model)
        top_model = layers.Dropout(0.25)(top_model)
        top_model = layers.Dense(label_num, activation='softmax')(top_model)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model