#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 8:48
@File:          MyConv1D.py
'''

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import InputSpec, Layer

def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    output_length = None
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

class MyConv1D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MyConv1D, self).__init__(**kwargs)
        self.supports_masking = True
        self.filters = filters
        self.kernel_size = (kernel_size, )
        self.strides = (strides, )
        self.padding = padding
        self.dilation_rate = (dilation_rate, )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, mask=None, **kwargs):
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format='channels_last',
            dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format='channels_last')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs * K.expand_dims(K.cast(self.compute_mask(inputs, mask=mask), K.dtype(outputs)), axis=2)

    def compute_output_shape(self, input_shape):
        new_dim = conv_output_length(
            input_shape[1],
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0])
        return (input_shape[0], new_dim, self.filters)

    def compute_mask(self, inputs, mask=None):
        new_dim = conv_output_length(
            K.int_shape(inputs)[1],
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0])
        return mask[:, :new_dim]

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MyConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))