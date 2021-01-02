#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 15:56
@File:          Add1D.py
'''

from keras import backend as K
from keras.layers import Layer

class Add1D(Layer):
    def __init__(self, **kwargs):
        super(Add1D, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        return sum(inputs) * K.expand_dims(K.cast(mask[0], K.dtype(inputs[0])), axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[0]