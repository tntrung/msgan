from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import partial
from modules.ops import *

'''
************************************************************************
* The small DC-GAN architecture for MNIST (28 x 28 x 1)
************************************************************************
'''

def encoder_dcgan_mnist(img, x_shape, z_dim=100, dim=64, \
                             kernel_size=5, stride=2, \
                             name = 'encoder', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, dim, kernel_size, stride))        #14 x 14 x dim
        y = conv_bn_relu(y, dim * 2, kernel_size, stride)  #7  x 7  x dim x 2
        y = conv_bn_relu(y, dim * 4, kernel_size, stride)  #4  x 4  x dim x 4
        logit = fc(y, z_dim)
        return logit
        
def generator_dcgan_mnist(z, x_shape, dim=64, kernel_size=5, stride=2, \
                          name = 'generator', \
                          reuse=True, training=True):
                           
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]  
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 4)                 
        y = tf.reshape(y, [-1, 4, 4, dim * 4])             #4 x 4 x dim x 4
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride) #8 x 8 x dim x 2
        # process the feature map 8x8 to to 7x7
        y = tf.reshape(y, [-1, 8 * 8 * 2 * dim])          
        y = relu(fc(y, 7 * 7 * 2 * dim))                   
        y = tf.reshape(y, [-1, 7, 7, 2 * dim])             #7 x 7 x dim x 2
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride) #14 x 14 x dim
        y = dconv(y, x_shape[2], kernel_size, stride)      #28 x 28 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

def discriminator_dcgan_mnist(img, x_shape, dim=64, \
                             kernel_size=5, stride=2, ss_task = 0,\
                             name='discriminator', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)
    
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))       #14 x 14 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride) #7  x 7  x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride) #4  x 4  x dim x 4
        feature = y
        logit = fc(y, 1)
        if ss_task == 1:
            k = 4
        elif ss_task == 2:
            k = 5
        else:
            k = -1        
        if ss_task > 0:
            print('[net_dcgan.py -- discriminator_dcgan_mnist] SS task = %d with k = %d classes' % (ss_task, k))
            cls   = fc(y, k)
            return tf.nn.sigmoid(logit),\
               logit,\
               tf.reshape(feature,[-1, 4 * 4 * dim * 4]), cls
        else:
            return tf.nn.sigmoid(logit),\
               logit,\
               tf.reshape(feature,[-1, 4 * 4 * dim * 4])

'''
************************************************************************
* The small DC-GAN architecture for CIFAR-10 and CIFAR-100 (32 x 32 x 3)
************************************************************************
'''

def encoder_dcgan_cifar(img, x_shape, z_dim=128, dim=64, kernel_size=5,\
                        stride=2, name = 'encoder', \
                        reuse=True, training=True):
                            
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))       #16 x 16 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride) #8 x 8 x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride) #4 x 4 x dim x 4
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride) #2 x 2 x dim x 8
        logit = fc(y, z_dim)
        return logit

def generator_dcgan_cifar(z, x_shape, dim=64, kernel_size=5, stride=2, \
                          name = 'generator', \
                          reuse=True, training=True):
                              
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)

    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 2 * 2 * dim * 8) 
        y = tf.reshape(y, [-1, 2, 2, dim * 8])              #2 x 2 x dim x 8
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride)  #4 x 4 x dim x 4
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)  #8 x 8 x dim x 2
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)  #16 x 16 x dim
        y = dconv(y, x_shape[2], kernel_size, stride)       #32 x 32 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

def discriminator_dcgan_cifar(img, x_shape, dim=64, kernel_size=5, \
                              stride=2, ss_task = 0, \
                              name='discriminator', \
                              reuse=True, training=True):
                                  
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, 2))             #16 x 16 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  #8 x 8 x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  #4 x 4 x dim x 4
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  #2 x 2 x dim x 8
        feature = y
        logit = fc(y, 1)
        if ss_task == 1:
            k = 4
        elif ss_task == 2:
            k = 5
        else:
            k = -1
        if ss_task > 0:
            print('[net_dcgan.py -- discriminator_dcgan_cifar] SS task = %d with k = %d classes' % (ss_task, k))
            cls   = fc(y, k)
            return tf.nn.sigmoid(logit), logit, \
                          tf.reshape(feature,[-1, 2 * 2 * dim * 8]), cls
        else:
            return tf.nn.sigmoid(logit), logit, \
                               tf.reshape(feature,[-1, 2 * 2 * dim * 8])


'''
************************************************************************
Unrolled-GAN network for Stacked-MNIST (28 x 28 x 3)
************************************************************************
'''

def encoder_dcgan_stacked_mnist(img, x_shape, z_dim=256, dim=8, \
                                kernel_size=5, stride=2, \
                                name = 'encoder', \
                                reuse=True, training=True):
                                 
    y0 = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    dconv_relu = partial(dconv,activation_fn=relu, biases_initializer=None)
    with tf.variable_scope(name, reuse=reuse):
        y1 =  relu(conv(y0, dim,  kernel_size, stride))    # 28 x 28 x 8
        y2 =  relu(conv(y1, dim * 2, kernel_size, stride)) # 14 x 14 x 16
        y3 =  relu(conv(y2, dim * 4, kernel_size, stride)) # 7 x 7 x 32
        y4 =  relu(conv(y3, dim * 8, kernel_size, stride)) # 4 x 4 x 64
        y5 =  tf.reshape(y4, [-1, 2 * 2 * dim * 8])
        z  =  fc(y5, z_dim)
        return z
        
def generator_dcgan_stacked_mnist(z, x_shape, dim=8, \
                                  kernel_size=5, stride=2, \
                                  name = 'generator', \
                                  reuse=True, training=True):
    
    # BN is important to make the model work.                   
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)                        
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]  
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])             #4 x 4 x 64
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride) #8 x 8 x 32
        y = y[:,:7,:7,:]                                   #7 x 7 x 32
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride) #14 x 14 x 16
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride) #28 x 28 x 8
        y = conv(y, x_shape[2], kernel_size, 1)            #28 x 28 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
                
def discriminator_dcgan_stacked_mnist(img, x_shape, dim=2, \
                                      kernel_size=5, stride=2, \
                                      name='discriminator', \
                                      ss_task = 0, reuse=True, \
                                      training=True):
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))   #2x14x14
        y = lrelu(conv(y, dim*2, kernel_size, stride)) #4x7x7
        y = lrelu(conv(y, dim*4, kernel_size, stride)) #8x4x4
        y = tf.reshape(y,[-1, 4 * 4 * dim * 4])
        feature = y
        logit = fc(y, 1)
        if ss_task == 1:
            k = 4
        elif ss_task == 2:
            k = 5
        else:
            k=-1
        if ss_task > 0:
            print('[net_dcgan.py -- discriminator_dcgan_cifar] SS task = %d with k = %d classes' % (ss_task, k))
            cls   = fc(y, k)
            return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[-1, int(4 * 4 * dim * 4)]), cls
        else:
            return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[-1,int( 4 * 4 * dim * 4)])
            
'''
************************************************************************
* The DC-GAN architecture for CIFAR-10 and CIFAR-100 (64 x 64 x 3)
************************************************************************
'''
def encoder_dcgan_celeba(img, x_shape, z_dim=128, dim=64, \
                              kernel_size=5, stride=2, \
                              name = 'encoder', reuse=True, training=True):
                                  
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))        #[32 x 32 x dim]
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  #[16 x 16 x 2 x dim]
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  #[8 x 8 x 4 x dim]
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  #[4 x 4 x 8 x dim]
        logit = fc(y, z_dim)                                #[z_dim]
        return logit

def generator_dcgan_celeba(z, x_shape, dim=64, \
                              kernel_size=5, stride=2, \
                              name = 'generator', reuse=True, training=True):
                                  
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)                 
        y = tf.reshape(y, [-1, 4, 4, dim * 8])             #4 x 4 x dim x 8
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride) #8 x 8 x dim x 4
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride) #16 x 16 x dim x 2
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride) #32 x 32 x dim
        y = dconv(y, x_shape[2], kernel_size, stride)      #64 x 64 x 3
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
               
def discriminator_dcgan_celeba(img, x_shape, dim=64, \
                                    kernel_size=5, stride=2, \
                                    ss_task = 0, name='discriminator', \
                                    reuse=True, training=True):
                                        
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, 2))            #32 x 32 x dim
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride) #16 x 16 x dim x 2
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride) #8 x 8 x dim x 4
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride) #4 x 4 x dim x 8
        feature = y
        logit = fc(y, 1)
        if ss_task == 1:
            k = 4
        elif ss_task == 2:
            k = 5
        else:
            k = -1
        if ss_task > 0:
            print('[net_dcgan.py -- discriminator_dcgan_celeba] SS task = %d with k = %d classes' % (ss_task, k))
            cls   = fc(y, k)
            return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1]), cls
        else:
            return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])
