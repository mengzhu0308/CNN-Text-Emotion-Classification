#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 16:28
@File:          cnn_model.py
'''

from keras import backend as K
from keras.layers import *

def res_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv1D(filters * 2, kernel_size, strides=strides, padding='same')(x)
    x = Lambda(lambda x_arg: x_arg[..., :filters] * K.sigmoid(x_arg[..., filters:]))(x)

    x = Conv1D(filters * 2, kernel_size, padding='same')(x)
    x = Lambda(lambda x_arg: x_arg[..., :filters] * K.sigmoid(x_arg[..., filters:]))(x)

    if K.int_shape(shortcut)[-1] == filters and strides == 1:
        x = add([shortcut, x])

    return x

def stack(x, filters, nblock, kernel_size=3, strides=2):
    x = res_block(x, filters, kernel_size=kernel_size, strides=strides)
    for _ in range(1, nblock):
        x = res_block(x, filters, kernel_size=kernel_size)

    return x

def CNN_Model(x, vocab_size, hidden_dim, max_length, num_classes=4):
    x = Embedding(vocab_size, hidden_dim, input_length=max_length)(x)
    x = Conv1D(32 * 2, 3, strides=2, padding='same')(x)
    x = Lambda(lambda x_arg: x_arg[..., :32] * K.sigmoid(x_arg[..., 32:]))(x)

    x = stack(x, 128, 2, strides=1)
    x = stack(x, 512, 2, strides=2)

    x = GlobalMaxPooling1D()(x)
    x = Dense(num_classes)(x)

    return x