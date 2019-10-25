from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import scipy
import numpy as np
import tensorflow as tf

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)


def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), \
        '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens

def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)
        
def load_checkpoint(checkpoint_dir, session, var_list=None):
    print('[mdutils.py -- load_checkpoint] Loading checkpoint: %s' % (checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print('[mdutils.py -- load_checkpoint] Checkpoint state: {}'.format(ckpt))
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        try:
            restorer = tf.train.Saver(var_list)
            restorer.restore(session, ckpt_path)
            print('[mdutils.py -- load_checkpoint] Loading successful! Copy variables from % s' % ckpt_path)
            return True
        except:
            print('[mdutils.py -- load_checkpoint] No suitable checkpoint from %s' % ckpt_path)
            return False
    return False
