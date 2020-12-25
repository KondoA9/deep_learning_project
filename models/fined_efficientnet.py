import tensorflow as tf
import tensorflow.keras.applications.efficientnet as net
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def _get_top_model(output, label_num):
    top_model = layers.GlobalAveragePooling2D()(output)
    top_model = layers.Dense(4096, activation='relu')(top_model)
    top_model = layers.Dropout(0.25)(top_model)
    top_model = layers.Dense(label_num, activation='softmax')(top_model)

    return top_model


class FinedEfficientNet:

    @staticmethod
    def buildB0(input_shape, label_num):
        efficientnet = net.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB1(input_shape, label_num):
        efficientnet = net.EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB2(input_shape, label_num):
        efficientnet = net.EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB3(input_shape, label_num):
        efficientnet = net.EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB4(input_shape, label_num):
        efficientnet = net.EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB5(input_shape, label_num):
        efficientnet = net.EfficientNetB5(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB6(input_shape, label_num):
        efficientnet = net.EfficientNetB6(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model

    @staticmethod
    def buildB7(input_shape, label_num):
        efficientnet = net.EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
        top_model = _get_top_model(efficientnet.output, label_num)
        
        model = Model(inputs=efficientnet.input, outputs=top_model)

        return model