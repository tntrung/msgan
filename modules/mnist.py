import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(source):
    mnist = input_data.read_data_sets(source)
    data  = mnist.train.images
    labels = mnist.train.labels
    labels = np.reshape(labels,(-1,1))
    print('[mnist.py] data shape: {}'.format(np.shape(data)))
    print('[mnist.py] labels shape: {}'.format(np.shape(labels)))
    return data, labels
    
def stacked_mnist_batch(data, batch_size):
    nb_data = np.shape(data)[0]
    ridx_ch1 = np.random.randint(nb_data, size=batch_size);
    ridx_ch2 = np.random.randint(nb_data, size=batch_size);
    ridx_ch3 = np.random.randint(nb_data, size=batch_size);
    mnist_ch1 = data[ridx_ch1, :]
    mnist_ch2 = data[ridx_ch2, :]
    mnist_ch3 = data[ridx_ch3, :]
    mnist_im1 = np.reshape(mnist_ch1, [batch_size, 28, 28, 1])
    mnist_im2 = np.reshape(mnist_ch2, [batch_size, 28, 28, 1])
    mnist_im3 = np.reshape(mnist_ch3, [batch_size, 28, 28, 1])
    batch = np.concatenate([mnist_im1, mnist_im2, mnist_im3], axis=3)
    batch = np.reshape(batch, [batch_size, 28 * 28 * 3])
    return batch
    
