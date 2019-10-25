# Dataset Utils
import os, sys
import glob
import math
import time

import numpy as np

from modules.imutils import imread

py_version = sys.version_info[0]

if py_version < 3:
    import cPickle as pickle
else:
    import pickle

""" Load pickle file """
def load_pickle(pklfile):
    fo = open(pklfile, 'rb')
    if py_version < 3:
       dict = pickle.load(fo) #python 2
    else:
       dict = pickle.load(fo, encoding='latin1') #python 3
    fo.close()
    return dict

""" List all dir with specific name """
def list_dir(folder_dir, ext="png"):
    all_dir = sorted(glob.glob(folder_dir+"*."+ext), key=os.path.getmtime)
    return all_dir
    
""" Preparing the data list of image datasets """
def prepare_image_list(source, ext='jpg', batch_size=64):
    # Count number of data images
    im_list  = list_dir(source, ext)
    nb_imgs  = len(im_list)
    nb_compl_batches  = int(math.floor(nb_imgs/batch_size))
    nb_total_batches  = nb_compl_batches
    if nb_imgs % batch_size != 0:
       nb_total_batches = nb_compl_batches + 1
    return im_list, nb_imgs, nb_compl_batches, nb_total_batches

""" Spliting the huge dataset into blocks """    
def datablock_preprocess(source, nb_splits = 32):
    # load block
    start = time.time()
    im_list  = list_dir(source, 'png')
    nb_imgs  = len(im_list)
    step = nb_imgs/nb_splits
    for nb_block in range(nb_splits):
        X_block = []
        for k in range(step*nb_block, step*(nb_block+1)):
            img = imread(im_list[k])
            print('[dbutils.py -- datablock_preprocess] loading: {}'.format(im_list[k]))
            img = img / 255.0
            X_block.append(np.reshape(img,(np.shape(img)[0] * np.shape(img)[1] * np.shape(img)[2])))
        X_block = np.array(X_block)
        np.save(source + "batch_"+str(nb_block)+".npy", X_block)
        print("[dbutils.py -- datablock_preprocess] Time to load images", time.time()-start)
