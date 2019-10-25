from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import partial
from modules.ops import *

'''
************************************************************************
* CNN architecture like (SN-GAN) for CIFAR-10 and CIFAR-100 (32 x 32 x 3)
* The number of feature maps of encoder and generator are doubled like 
* in SN-GAN.
************************************************************************
'''

def encoder_sngan_cifar(img, x_shape, z_dim=128, dim=64, \
                        kernel_size=4, stride=2, \
                        name = 'encoder', reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, dim * 2, 3, 1))                         #32 x 32 x dim (x 2)
        y = conv_bn_lrelu(y, dim * 2 * 2, kernel_size, stride)   #16 x 16 x dim x 2 (x 2)
        y = conv_bn_lrelu(y, dim * 4 * 2, kernel_size, stride)   #8 x 8 x dim x 4 (x 2)
        y = conv_bn_lrelu(y, dim * 8 * 2, kernel_size, stride)   #4 x 4 x dim x 8 (x 2)
        logit = fc(y, z_dim)
        return logit

def generator_sngan_cifar(z, x_shape, dim=64, \
                          kernel_size=4, stride=2, \
                          name = 'generator', reuse=True, training=True):
                              
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)   
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    z_dim = z.get_shape()[1].value
    assert(z_dim == 128) # check latent dim = 128 like SN-GAN
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(z, 4 * 4 * dim * 8 * 2))                   
        y = tf.reshape(y, [-1, 4, 4, dim * 8 * 2])             #4 x 4 x dim x 8 (x 2)
        y = dconv_bn_relu(y, dim * 4 * 2, kernel_size, stride) #8 x 8 x dim x 4 (x 2)
        y = dconv_bn_relu(y, dim * 2 * 2, kernel_size, stride) #16 x 16 x dim x 4 (x 2)
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)     #32 x 32 x dim (x 2)
        y = conv(y, x_shape[2], 3, 1)                          #32 x 32 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

def discriminator_sngan_cifar(img, x_shape, dim=64,\
                              kernel_size=4, stride=2, ss_task = 0, \
                              name='discriminator', \
                              reuse=True, training=True):
                                       
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size - 1, stride - 1)) 
        y = lrelu(conv(y, dim, kernel_size, stride))
        y = lrelu(conv(y, dim * 2, kernel_size - 1, stride - 1))
        y = lrelu(conv(y, dim * 2, kernel_size, stride))
        y = lrelu(conv(y, dim * 4, kernel_size - 1, stride - 1))
        y = lrelu(conv(y, dim * 4, kernel_size, stride))
        y = lrelu(conv(y, dim * 8, kernel_size - 1, stride - 1))
        feature = y
        logit = fc(y, 1)
        if ss_task == 1:
            k = 4
        elif ss_task == 2:
            k = 5
        else:
            k = -1
        print('[net_dcgan.py -- discriminator_sngan_cifar] SS task = %d with k = %d classes' % (ss_task, k))
        if ss_task > 0:
            cls   = fc(y, k)
            return tf.nn.sigmoid(logit), logit, \
                         tf.reshape(feature,[img.get_shape().as_list()[0], -1]), cls
        else:
            return tf.nn.sigmoid(logit), logit, \
                              tf.reshape(feature,[img.get_shape().as_list()[0], -1])
