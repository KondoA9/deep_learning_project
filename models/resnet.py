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

def _ResNetLayer(input, filter_kernel_shape_list, num, final_block=False):
    x = None

    l_num = len(filter_kernel_shape_list)

    for i, l in enumerate(filter_kernel_shape_list):
        for j in range(num):
            if i == 0 and j == 0:
                x = _ResidualBlock(input, l[0], l[1], downsample=True, final_block=final_block)
            elif i == l_num - 1 and j == num - 1:
                x = _ResidualBlock(x, l[0], l[1], downsample=True, final_block=final_block)
            else:
                x = _ResidualBlock(x, l[0], l[1], downsample=True, final_block=final_block)

    return x

class ResNet:
    
    @staticmethod
    def build(input_shape, label_num):
        #first layer
        inputs_layer = layers.Input(shape=input_shape)
        output = layers.Conv2D(64, 7, strides=2)(inputs_layer)
        output = layers.MaxPool2D(pool_size=(3, 3), strides=2)(output)

        l = [
            (64, 1),
            (64, 3),
            (256, 1)
        ]
        x = _ResNetLayer(output, l, 3)

        l = [
            (128, 1),
            (128, 3),
            (512, 1)
        ]
        x = _ResNetLayer(output, l, 4)

        l = [
            (256, 1),
            (256, 3),
            (1024, 1)
        ]
        x = _ResNetLayer(output, l, 6)

        l = [
            (512, 1),
            (512, 3),
            (2048, 1)
        ]
        x = _ResNetLayer(output, l, 3, final_block=True)

        '''
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
        '''

        #dense
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

        predictions = layers.Dense(label_num, activation='softmax')(x)

        return Model(inputs=inputs_layer, outputs=predictions)
