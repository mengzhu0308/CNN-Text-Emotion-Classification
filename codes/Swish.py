#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 15:43
@File:          Swish.py
'''

from keras import backend as K
import tensorflow as tf
from keras.layers import Layer

class Swish(Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        return tf.nn.swish(inputs) * K.expand_dims(K.cast(mask, K.dtype(inputs)), axis=2)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape