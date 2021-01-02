#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:34
@File:          MyBidirectional.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class MyBidirectional(Layer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(MyBidirectional, self).__init__(**args)
        self.supports_masking = True
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs, mask=None, **kwargs):
        mask = K.expand_dims(K.cast(mask, K.dtype(inputs)), axis=2)
        x_forward = self.forward_layer(inputs)
        x_backward = self.reverse_sequence(inputs, mask)
        x_backward = self.backward_layer(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)

        return x * mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.forward_layer.units * 2,)

    def compute_mask(self, inputs, mask=None):
        return mask