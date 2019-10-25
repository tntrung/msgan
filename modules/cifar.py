import os, sys
import shutil
import numpy as np

from modules.dbutils import *
from modules.fiutils import *

# unpickle for cifar datasets
def unpickle(myfile, dbname = 'cifar10'):
    exists = os.path.exists(myfile)
    if exists:
        dict = load_pickle(myfile)
        #print('[cifar.py] dict keys = : {}'.format(dict.keys()))
        if dbname in ['cifar10']:
            return dict['data'], dict['labels']
        elif dbname in ['cifar100']:
            return dict['data'], dict['fine_labels']
    else:
        print('[cifar.py - unpickle] %s is invalid' % (myfile))
        exit()
        
# load cifar10        
def load_cifar10(data_dir, filenames):
    
    # check and download data first
    flag = os.path.exists(os.path.join(data_dir,filenames[0]))
    if flag == False:
        # download and unzip cifar data
        filepath = download('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', data_dir);
        decompress(filepath, data_dir);
        # cifar-10-python.tar.gz is extracted into 'cifar-10-batches-py'
        # copy the files out and delete the folder
        decom_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        copy_all_files(decom_dir, data_dir)
        print('[cifar10.py - load_cifar10] removing %s' % (decom_dir))
        remove_dir(decom_dir)
        os.remove(os.path.join(data_dir, 'cifar-10-python.tar.gz'))
        
    all_data = []
    all_labels = []
    for filename in filenames:          
        data, labels = unpickle(os.path.join(data_dir,filename), 'cifar10')
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    #normalize into [0,1]
    images = images.astype(float)/255.0
    #normalize into [-1,1]
    #images = (images.astype(float)/255.0 - 0.5) * 2
    images = np.reshape(images,(-1, 3, 32, 32))
    images = np.transpose(images,(0,2,3,1)) #tranpose to standard order of channels
    images = np.reshape(images,(-1, 32*32*3))
    
    print('[cifar.py - load_cifar10] data shape: {}'.format(np.shape(images)))
    print('[cifar.py - load_cifar10] label shape: {}'.format(np.shape(labels)))
    return images, labels

# load cifar100       
def load_cifar100(data_dir, filenames):
	
	# check and download data first
    flag = os.path.exists(os.path.join(data_dir,filenames[0]))
    if flag == False:
        # download and unzip cifar data
        filepath = download('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', data_dir);
        decompress(filepath, data_dir);
        # cifar-10-python.tar.gz is extracted into 'cifar-10-batches-py'
        # copy the files out and delete the folder
        decom_dir = os.path.join(data_dir, 'cifar-100-python')
        copy_all_files(decom_dir, data_dir)
        print('[cifar10.py - load_cifar10] removing %s' % (decom_dir))
        remove_dir(decom_dir)
        os.remove(os.path.join(data_dir, 'cifar-100-python.tar.gz'))
        
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(os.path.join(data_dir,filename), 'cifar100')
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    #normalize into [0,1]
    images = images.astype(float)/255.0
    #normalize into [-1,1]
    #images = (images.astype(float)/255.0 - 0.5) * 2
    images = np.reshape(images,(-1, 3, 32, 32))
    images = np.transpose(images,(0,2,3,1)) #tranpose to standard order of channels
    images = np.reshape(images,(-1, 32*32*3))
    
    print('[cifar.py - load_cifar100] data shape: {}'.format(np.shape(images)))
    print('[cifar.py - load_cifar100] label shape: {}'.format(np.shape(labels)))
    return images, labels    
