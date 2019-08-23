import tensorflow as tf
import numpy as np


def weight_variable(shape, weight_initializer):
    return tf.get_variable(name='weights', shape=shape, dtype=tf.float32,
                           initializer=weight_initializer)


def bias_variable(shape, bias_initializer):
    return tf.get_variable(name='bias', shape=shape, dtype=tf.float32,
                           initializer=bias_initializer)


def conv2d(inputs, output_channels, kernel_size, stride=1, padding='SAME', **kwargs):
    weight_initializer = kwargs.pop('weight_initializer', tf.random_normal_initializer(mean=0.0, stddev=.01))
    bias_initializer = kwargs.pop('bias_initializer', tf.constant_initializer(value=1.0))
    in_channels = int(inputs.get_shape()[-1])
    filter = weight_variable(shape=[kernel_size, kernel_size, in_channels, output_channels],
                             weight_initializer=weight_initializer)
    bias = bias_variable(shape=[output_channels], bias_initializer=bias_initializer)
    return tf.nn.conv2d(input=inputs, filter=filter, strides=[1, stride, stride, 1], padding=padding) + bias


def lrn(inputs, depth_radius=5, bias=2, alpha=1e-4, beta=0.75):
    return tf.nn.local_response_normalization(input=inputs,
                                              depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)


def max_pool2d(inputs, kernel_size, stride, padding='SAME', **kwargs):
    return tf.nn.max_pool(value=inputs, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def fully_connected(inputs, num_outputs, **kwargs):
    weight_initializer = kwargs.pop('weight_initializer', tf.random_normal_initializer(mean=0.0, stddev=0.01))
    bias_initializer = kwargs.pop('bias_initializer', tf.constant_initializer(value=1.0))
    num_inputs = int(inputs.get_shape()[-1])
    weight = weight_variable([num_inputs, num_outputs], weight_initializer=weight_initializer)
    bias = bias_variable([num_outputs], bias_initializer=bias_initializer)
    return tf.matmul(inputs, weight) + bias


def flatten(inputs, **kwargs):
    dims = int(np.prod(inputs.get_shape()[1:]))
    return tf.reshape(inputs, [-1, dims])
