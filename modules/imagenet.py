import os, sys

from modules.fiutils import *
from modules.dbutils import *

# load imagenet 32x32
def load_imagenet_32(data_dir, nb_splits = 32):
    # check and download data first
    flag = os.path.exists(os.path.join(data_dir))
    if flag == False:
        # download and unzip cifar data
        filepath = download('http://image-net.org/small/train_32x32.tar', data_dir);
        #filepath = data_dir + 'train_32x32.tar'
        decompress(filepath, data_dir);
        # train_32x32.tar is extracted into 'train_32x32'
        # copy the files out and delete the folder
        decom_dir = os.path.join(data_dir, 'train_32x32')
        copy_all_files(decom_dir, data_dir)
        print('[imagenet.py - load_imagenet_32] removing %s' % (decom_dir))
        remove_dir(decom_dir)
        os.remove(os.path.join(data_dir, 'train_32x32.tar'))
        
        # preprocessing
        datablock_preprocess(data_dir, nb_splits)
