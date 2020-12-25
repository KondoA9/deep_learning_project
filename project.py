from tensorflow.keras import callbacks, utils

from .models.resnet import ResNet
from .models.fined_resnet import FinedResNet
from .models.fined_efficientnet import FinedEfficientNet

import os
import sys


class Project:

    callbacks = [None] * 0

    def __init__(self, name, tb_update_freq='batch', lr_setter=None):
        self.name = name

        # directories
        self._tensorboard_dir = "../tmp/tb/" + name + '/'
        self._checkpoint_dir = "../tmp/cp/" + name + '/'
        self._model_image_dir = "../model_image/"
        self._checkpoint_path = self._checkpoint_dir + 'checkpoint'

        # callbacks
        self.callbacks.append(callbacks.ModelCheckpoint(
            filepath=self._checkpoint_path,
            save_weights_only=True,
            verbose=1))

        self.callbacks.append(callbacks.TensorBoard(
            log_dir=self._tensorboard_dir, histogram_freq=1,
            update_freq=tb_update_freq))

        if not lr_setter == None:
            self.callbacks.append(callbacks.LearningRateScheduler(
                lr_setter,
                verbose=0))

        # create directories
        if not os.path.exists(self._tensorboard_dir):
            os.makedirs(self._tensorboard_dir, exist_ok=True)
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        if not os.path.exists(self._model_image_dir):
            os.makedirs(self._model_image_dir, exist_ok=True)

    def _exist_ckpt(self):
        return os.path.exists(self._checkpoint_path)

    def plot_model(self, model):
        utils.plot_model(model,
                         self._model_image_dir + self.name + '.png',
                         show_shapes=True)

    def load_ckpt(self, model):
        if self._exist_ckpt():
            print("Load checkpoint: " + self._checkpoint_path)
            model.load_weights(self._checkpoint_path)
        else:
            print("Checkpoint is not found: " + self._checkpoint_path)

        return model

    def build(self, model_name, input_shape, label_num):
        """
        Available model_name:
            'resnet',
            'fined_resnet'
            'fined_efficientnet'

        """

        shape = (input_shape[0], input_shape[1], 3)

        if model_name == 'resnet':
            return ResNet.build(shape, label_num)

        elif model_name == 'fined_resnet':
            return FinedResNet.build(shape, label_num)

        elif model_name == 'fined_efficientnet':
            return FinedEfficientNet.build(shape, label_num)

        else:
            print("Unknown model")
            sys.exit()
