from tensorflow.keras import layers
from tensorflow.keras.models import Model

def _Conv2DBlock(input, filters, kernels, dropout=False):
    x = layers.BatchNormalization()(input)
    x = layers.ReLU()(x)
    if dropout == True:
        x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(filters, kernels, padding='same')(x)
    return x

def _ResidualBlock(input, filters, kernels, downsample=False, final_block=False):
    x = _Conv2DBlock(input, filters, kernels)
    x = _Conv2DBlock(x, filters, kernels, dropout=final_block)

    y = input if downsample == False else layers.Conv2D(
        filters, kernels, padding='same')(input)

    output = layers.add([x, y])

    return output

class ResNet:
    
    @staticmethod
    def build(input_shape, label_num):
        #first layer
        inputs_layer = layers.Input(shape=input_shape)
        output = layers.Conv2D(64, 3)(inputs_layer)
        output = layers.MaxPool2D()(output)

        #conv_1
        x = _ResidualBlock(output, 64, 3)
        x = _ResidualBlock(x, 64, 3)
        x = _ResidualBlock(x, 64, 3)

        #conv_2
        x = _ResidualBlock(x, 128, 3, True)
        x = _ResidualBlock(x, 128, 3)
        x = _ResidualBlock(x, 128, 3)

        #conv_3
        x = _ResidualBlock(x, 256, 3, True)
        x = _ResidualBlock(x, 256, 3)
        x = _ResidualBlock(x, 256, 3)

        #conv_4
        x = _ResidualBlock(x, 512, 3, True)
        x = _ResidualBlock(x, 512, 3)
        x = _ResidualBlock(x, 512, 3, final_block=True)

        #dense
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

        predictions = layers.Dense(label_num, activation='softmax')(x)

        return Model(inputs=inputs_layer, outputs=predictions)
