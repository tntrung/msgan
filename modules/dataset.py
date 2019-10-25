import math
import os
import sys

import numpy as np
import tensorflow as tf

from modules.mnist    import load_mnist, stacked_mnist_batch
from modules.celeba   import preprocess, load_celeba
from modules.cifar    import load_cifar10, load_cifar100
from modules.stl10    import load_stl10
from modules.imagenet import load_imagenet_32
from modules.imutils  import imread
from modules.dbutils  import list_dir, prepare_image_list

class Dataset(object):

    def __init__(self, name='mnist', source='./data/mnist/', batch_size = 64, seed = 0):

        self.name            = name
        self.source          = source
        self.batch_size      = batch_size
        self.seed            = seed
        np.random.seed(seed) # To make your "random" minibatches the same for experiments

        self.count           = 0

        # the dimension of vectorized and orignal data samples
        if self.name == 'mnist':
            self.data_vec_dim = 784   #28x28
            self.data_origin_shape = [28, 28, 1]
        elif self.name in ['mnist-1k']:
            self.data_vec_dim = 2352  #28x28x3
            self.data_origin_shape = [28, 28, 3]
        elif self.name in ['cifar10', 'cifar100' , 'imagenet_32']:
            self.data_vec_dim = 3072  #32x32x3
            self.data_origin_shape = [32, 32, 3]
            self.nb_splits = 32
        elif self.name == 'celeba':
            self.data_vec_dim = 12288 #64x64x3
            self.data_origin_shape = [64, 64, 3]
        elif self.name == 'stl10':
            self.data_vec_dim = 6912  # 48x48x3
            self.data_origin_shape = [48, 48, 3]
        else:
            self.data_vec_dim = 0     #0
            self.data_origin_shape = [0, 0, 0]
            print('[dataset.py - __init__] dbname = %s is unrecognized.\n' % (self.name))
            exit()
        
        tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

        if name == 'mnist':
            self.data, self.labels = load_mnist(self.source)
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'mnist-1k':
            self.data_mnist, self.labels_mnist = load_mnist(self.source)
            self.nb_mnist = np.shape(self.data_mnist)[0]
        elif name == 'cifar10':
            data_files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
            self.data, self.labels = load_cifar10(source, data_files)
            self.labels = np.reshape(self.labels,(-1,1))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'cifar100':
            data_files = ['train']
            self.data, self.labels = load_cifar100(source, data_files)
            self.labels = np.reshape(self.labels,(-1,1))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)    
        elif name == 'celeba':
            load_celeba(self.source)
            self.im_list, \
            self.nb_imgs, \
            self.nb_compl_batches, \
            self.nb_total_batches = prepare_image_list(source, 'jpg', self.batch_size)
            self.count = 0
            self.color_space = 'RGB'
        elif name == 'stl10':
            load_stl10(self.source)
            self.im_list, \
            self.nb_imgs, \
            self.nb_compl_batches, \
            self.nb_total_batches = prepare_image_list(source, 'png', self.batch_size)
            self.count = 0
            self.color_space = 'RGB'
        elif name == 'imagenet_32':
            load_imagenet_32(self.source, self.nb_splits)
            self.minibatches = self.random_mini_batches([], self.batch_size, self.seed)
            print('[dataset.py -- __init__] The number of minibatches = %s' % (len(self.minibatches)))
            #self.im_list, \
            #self.nb_imgs, \
            #self.nb_compl_batches, \
            #self.nb_total_batches = prepare_image_list(source, 'jpg', self.batch_size)
            self.count = 0
            #self.color_space = 'RGB'
        
    def db_name(self):
        return self.name
        
    def db_source(self):
        return self.source    

    def data_dim(self):
        return self.data_vec_dim
        
    def data_shape(self):
        return self.data_origin_shape
                    
    def mb_size(self):
        return self.batch_size

    def next_batch(self):

        if self.name == 'mnist' or self.name == 'cifar10' or self.name =='cifar100':
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
            batch = self.minibatches[self.count]
            self.count = self.count + 1
            return batch.T        

        # if self.name == 'mnist' or self.name == 'cifar10': #or self.name == 'stl10':
        #     if self.count == len(self.minibatches):
        #         self.count = 0
        #         self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        #     batch = self.minibatches[self.count]
        #     self.count = self.count + 1
        #     return batch.T
        elif self.name in ['mnist-1k']:
            batch = stacked_mnist_batch(self.data_mnist, self.batch_size)
            return batch
        elif self.name in ['celeba', 'stl10']:
            batch = self.random_mini_batches([], self.batch_size, self.seed)
            return batch
        elif self.name in ['imagenet_32']:
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches = self.random_mini_batches([], self.batch_size, self.seed)
                print('[dataset.py -- next_batch] The number of minibatches = %s' % (len(self.minibatches)))
            batch = self.minibatches[self.count]
            self.count = self.count + 1
            #print('count = %d' % (self.count))
            return batch
               

    def fixed_mini_batches(self, mini_batch_size = 64):
        if self.name == 'mnist' or self.name == 'cifar10' or self.name == 'cifar100':
            X = self.data
            Y = self.labels
            m = X.shape[0]
            mini_batches = []
            mini_labels  = []
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch = X[k * self.batch_size : (k+1) * self.batch_size, :]
                mini_label = Y[k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch)
                mini_labels.append(mini_label)

            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch = X[num_complete_minibatches * self.batch_size : m, :]
            #    mini_label = Y[num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch)
            #    mini_labels.append(mini_label)
            
            return mini_batches, mini_labels
            
    # Random minibatches for training
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        if self.name in ['mnist', 'cifar10', 'cifar100']:
            m = X.shape[1]    # number of training examples
            mini_batches = []
            mini_labels  = []
                            
            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation]

            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch_X)
                mini_batch_Y = shuffled_Y[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_labels.append(mini_batch_Y)
            
            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch_X = shuffled_X[:, num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch_X)
            
            return mini_batches, mini_labels
            
        elif self.name in ['celeba', 'stl10']:
            
            #print('[dataset.py - random_mini_batches] count = %d' % (self.count))
            if self.count == 0:
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]  
            elif self.count > 0 and self.count < self.nb_compl_batches:
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                           
            elif self.count == self.nb_compl_batches and self.nb_total_batches > self.nb_compl_batches:
            #    cur_batch = self.permutation[self.nb_compl_batches * self.batch_size : self.nb_imgs]
            #elif self.count >= self.num_total_batches:
                self.count = 0
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                
            else:
                print('[dataset.py - random_mini_batches] something is wrong with mini-batches')
            
            mini_batches = []
            #print('cur_batch: {}'.format(cur_batch))

            # handle complete cases
            for k in cur_batch:
                img = imread(self.im_list[k])
                if self.name == 'celeba':
                    img = preprocess(img)
                if self.color_space == 'YUV':
                    img = RGB2YUV(img)
                img = img / 255.0
                mini_batches.append(np.reshape(img,(1,np.shape(img)[0] * np.shape(img)[1] * np.shape(img)[2])))
            #print('merging shape', np.shape(mini_batches))
            mini_batches = np.concatenate(mini_batches, axis=0)
            self.count = self.count + 1
                    
            return mini_batches
            
        elif self.name in ['imagenet_32']:
             
            batch_index = np.random.randint(self.nb_splits, size=1)[0]
            np_file = self.source + "/batch_"+str(batch_index)+".npy"
            print('[dataset.py -- random_mini_batches] processing block: %s for mini-batches generation' % (np_file))
            X_1 = np.load(np_file)
            m = X_1.shape[0] # number of training examples
            mini_batches = []
            permutation = list(np.random.permutation(m))
            shuffled_X = X_1[permutation]
            self.num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, self.num_complete_minibatches):
                mini_batch_X = shuffled_X[k * self.batch_size : (k+1) * self.batch_size]
                mini_batch_X = np.reshape(mini_batch_X, (self.batch_size, 3 * 32 * 32))
                # for img in mini_batch_X:
                #     imwrite(np.reshape(img,(32,32,3)), "./image.jpg")
                #     exit()
                mini_batches.append(mini_batch_X)
            return mini_batches
            
    def load_test(self):
        
        if self.name == 'cifar10':
           data_files = ['test_batch']
           self.data, _ = load_cifar10(self.source, data_files)
           return self.data
