#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 16:28
@File:          CNN_model.py
'''

from keras import backend as K
from keras.layers import *

from MyConv1D import MyConv1D
from Add1D import Add1D
from Swish import Swish

def res_block(x, filters, kernel_size=5, strides=1):
    shortcut = x
    x = MyConv1D(filters, 1)(x)
    x = Swish()(x)

    x = MyConv1D(filters, kernel_size, strides=strides, padding='same')(x)
    x = Swish()(x)

    x = MyConv1D(filters * 4, 1)(x)
    x = Swish()(x)

    if K.int_shape(shortcut)[-1] == filters * 4 and strides == 1:
        x = Add1D()([shortcut, x])

    return x

def stack(x, filters, nblock, kernel_size=5, strides=1):
    x = res_block(x, filters, kernel_size=kernel_size, strides=strides)
    for _ in range(1, nblock):
        x = res_block(x, filters, kernel_size=kernel_size)

    return x

def CNN_Model(x, vocab_size, max_length, hidden_dim=64, num_classes=4):
    x = Embedding(vocab_size, hidden_dim, input_length=max_length, mask_zero=True)(x)
    x = MyConv1D(hidden_dim, 9, padding='same')(x)
    x = Swish()(x)

    x = stack(x, hidden_dim, 2, strides=2)
    x = stack(x, hidden_dim * 2, 2, strides=2)
    x = stack(x, hidden_dim * 4, 2, strides=2)

    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes)(x)

    return x