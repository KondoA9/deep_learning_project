from tensorflow.keras import callbacks, utils

from .models.resnet import ResNet
from .models.fined_resnet import FinedResNet

import os
import sys


class Project:

    callbacks = [None] * 0

    def __init__(self, name, tb_update_freq='batch', lr_setter=None):
        self.name = name

        # directories
        self._tensorboard_dir = "../tmp/tb/" + name + '/'
        self._checkpoint_dir = "../tmp/cp/" + name + '/'
        self._checkpoint_path = self._checkpoint_dir + 'checkpoint'

        # callbacks
        self.callbacks.append(callbacks.ModelCheckpoint(
            self._checkpoint_path, save_weights_only=True, verbose=1))

        self.callbacks.append(callbacks.TensorBoard(
            log_dir=self._tensorboard_dir, histogram_freq=1,
            update_freq=tb_update_freq))

        if not lr_setter == None:
            self.callbacks.append(callbacks.LearningRateScheduler(
                lr_setter,
                verbose=0))

        # create directories
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def _exist_ckpt(self):
        return os.path.exists(self._checkpoint_path)

    def plot_model(self, model):
        utils.plot_model(model,
                         '../model_image/' + self.name + '.png',
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

        """

        shape = (input_shape[0], input_shape[1], 3)

        if self.name == 'resnet':
            return ResNet.build(shape,label_num)

        elif self.name == 'fined_resnet':
            return FinedResNet.build(shape, label_num)

        else:
            print("Unknown model")
            sys.exit()
