from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

'''
************************************************************************
* Pre-defined fully connected layers
************************************************************************
'''
def flatten_fully_connected(inputs,
                            num_outputs,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):
    with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)

# lrelu
def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y



'''
************************************************************************
* Function alias
************************************************************************
'''

conv = partial(slim.conv2d, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
             
dconv = partial(slim.conv2d_transpose, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
             
fc = partial(flatten_fully_connected, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
                                                                     
batch_norm = partial(slim.batch_norm, \
          decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
          
relu  = tf.nn.relu
lrelu = partial(leak_relu, leak=0.2)
