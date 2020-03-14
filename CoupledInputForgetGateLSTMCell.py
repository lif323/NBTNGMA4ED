#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.util.tf_export import keras_export

class CoupledInputForgetGateLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        # lstm 维度
        self.units = units
        super(CoupledInputForgetGateLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(shape=(input_dim, self.units * 4), name='kernel',
            initializer=initializers.get('glorot_uniform'))

        self.u = self.add_weight(shape=(self.units, self.units * 4),
                                                name='recurrent_kernel',
                                                initializer=initializers.get('orthogonal'))
        self.bias = self.add_weight(
            shape=(self.units * 4), name='bias',
            initializer=initializers.get('zeros'))

        self.recurrent_activation = activations.get('hard_sigmoid')
        self.activation = activations.get('tanh')

    def call(self, inputs, states):
        last_h = states[0]
        last_c = states[1]
        w_i, w_f, w_c, w_o = tf.split(self.w, num_or_size_splits=4, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
        # w x
        x_i = K.dot(inputs, w_i)
        x_f = K.dot(inputs, w_f)
        x_c = K.dot(inputs, w_c)
        x_o = K.dot(inputs, w_o)
        # w x + b
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

        u_i, u_f, u_c, u_o = tf.split(self.u, num_or_size_splits=4, axis=1)
        # w x + u * h + x
        i = self.recurrent_activation(x_i + K.dot(last_h, u_i))
        f = self.recurrent_activation(x_f + K.dot(last_h, u_f))
        c = (1 - i) * last_c + self.activation(x_c + K.dot(last_h, u_c))
        o = self.recurrent_activation(x_o + K.dot(last_h, u_o))

        # 计算 h
        h = o * self.activation(c)
        return h, (h, c)

class Rnn(tf.keras.layers.Layer):
    def __init__(self, units=128):
        super(Rnn, self).__init__()
        self.cell = CoupledInputForgetGateLSTMCell(units)
        self.init_state = None
    def build(self, input_shape):
        shape = input_shape.as_list()
        n_batch = shape[0]
        init_h = tf.zeros(shape=[n_batch, self.cell.units])
        init_c = init_h
        self.init_state = (init_h, init_c)

    def call(self, inputs):
        # time step
        ts = inputs.shape.as_list()[1]
        h, c = self.init_state
        for i in range(ts):
            h, (h, c) = self.cell(inputs[:, i], (h, c))
        return h

if __name__ == "__main__":
    a = tf.random.normal(shape=(2, 3, 4))
    rnn = Rnn()
    h = rnn(a)
    print(h.shape)
