import numpy as np
import random
import math
from   skimage import io, img_as_ubyte
import tensorflow as tf
import copy
import scipy

'''
************************************************************************
* IO image
************************************************************************
'''

def imread(path, is_grayscale=False):
    img = scipy.ndimage.imread(path).astype(np.float)
    return np.array(img)
    
def imwrite(image, path):
    if image.ndim == 3 and image.shape[2] == 1: # grayscale images
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return io.imsave(path, img_as_ubyte(image))

def immerge_row_col(N):
    c = int(np.floor(np.sqrt(N)))
    for v in range(c,N):
        if N % v == 0:
            c = v
            break
    r = N / c
    return r, c
    
def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    @images: is in shape of N * H * W(* C=1 or 3)
    """
    row = int(row)
    col = int(col)
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image
    return img
    
def imsave_batch(X, data_shape, im_save_path):
    im_save = np.reshape(X,(-1, data_shape[0], data_shape[1], data_shape[2]))
    ncols, nrows = immerge_row_col(np.shape(im_save)[0])
    im_merge = immerge(im_save, ncols, nrows)
    imwrite(im_merge, im_save_path)

'''
************************************************************************
* Vec/Image utils
************************************************************************
'''
def tf_img_to_vec(X, data_shape):
    return tf.reshape(X,[-1,data_shape[0]*data_shape[1]*data_shape[2]])
    
def tf_vec_to_img(X, data_shape):
    return tf.reshape(X,[-1,data_shape[0],data_shape[1],data_shape[2]])                    	
'''
************************************************************************
* Rotation based Self-supervised Tasks
************************************************************************
'''

def tf_argument_image_rotation(X, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf_vec_to_img(X, data_shape)
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg = tf_img_to_vec(Xarg, data_shape)
    
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

'''
Rotation of samples with additional fake class
'''
def tf_argument_image_rotation_plus_fake(X, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf_vec_to_img(X, data_shape)
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg = tf_img_to_vec(Xarg, data_shape)
        
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

'''
Rotation of mixture of real/fake samples with additional fake class
'''
def tf_argument_image_rotation_and_fake_mix(X, X_f, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp      = tf_vec_to_img(X, data_shape)
    Xtmp_fake = tf_vec_to_img(X_f, data_shape)
    n         = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    v_4  = tf.constant([[0., 0., 0., 0., 1.]])
    X_4  = Xtmp_fake[:n,:,:,:]
    l_4  = tf.tile(v_4, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3, X_4], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3, l_4], axis=0)
    
    Xarg = tf_img_to_vec(Xarg, data_shape)
        
    if ridx is None:
        ridx = tf.range(0,nimgs*5,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx
