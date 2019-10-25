from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

class classify:
    def __init__(self):
        self.Xmn = tf.placeholder(tf.float32, [None, 784])
        self.Ymn = tf.placeholder(tf.float32, [None, 10])

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def Build_model(self):
        with tf.name_scope('reshape'):
            self.x_image = tf.reshape(self.Xmn, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_variable([5, 5, 1, 32])
            self.b_conv1 = self.bias_variable([32])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_variable([5, 5, 32, 64])
            self.b_conv2 = self.bias_variable([64])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            self.b_fc1 = self.bias_variable([1024])

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_variable([1024, 10])
            self.b_fc2 = self.bias_variable([10])

            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        return self.y_conv, self.keep_prob

    def Train(self,data_source,save_path="Classifier/model.ckpt"):
        mnist = input_data.read_data_sets(data_source, one_hot=True)
        self.y_conv, self.keep_prob = self.Build_model()
        saver = tf.train.Saver()
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Ymn,
                                                                logits=self.y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.Ymn, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        self.sess_classifier = tf.Session()
        self.sess_classifier.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.sess_classifier.run(accuracy, feed_dict={self.Xmn: batch[0], self.Ymn: batch[1], self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.sess_classifier.run(train_step, feed_dict={self.Xmn: batch[0], self.Ymn: batch[1], self.keep_prob: 0.5})
        print('Finish Training Process')
        test_ = mnist.test.next_batch(1000)
        test_accuracy = self.sess_classifier.run(accuracy, feed_dict = {self.Xmn: test_[0], self.Ymn: test_[1], self.keep_prob:1.0})
        print('Test accuracy %g' %(test_accuracy))
        save_path = saver.save(sess=self.sess_classifier, save_path=save_path)
        print('Model saved in file: %s'%save_path)

    def TrainwithoutSave(self, data_source):
        mnist = input_data.read_data_sets(data_source, one_hot=True)
        self.y_conv, self.keep_prob = self.Build_model()
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Ymn,
                                                                logits=self.y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.Ymn, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        self.sess_classifier = tf.Session()
        self.sess_classifier.run(tf.global_variables_initializer())
        for i in range(20000): #20000
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.sess_classifier.run(accuracy, feed_dict={self.Xmn: batch[0], self.Ymn: batch[1], self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.sess_classifier.run(train_step, feed_dict={self.Xmn: batch[0], self.Ymn: batch[1], self.keep_prob: 0.5})
        print('Finish Training Process')
        test_ = mnist.test.next_batch(1000)
        test_accuracy = self.sess_classifier.run(accuracy, feed_dict = {self.Xmn: test_[0], self.Ymn: test_[1], self.keep_prob:1.0})
        print('Test accuracy %g' %(test_accuracy))

    def Evaluate_Labels(self, Images, model_path="Classifier/model.ckpt"):
        self.Build_model()
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        sess_classifier = tf.Session(config=run_config)
        saver = tf.train.Saver()
        saver.restore(sess_classifier, save_path=model_path)
        Curr_Preds = sess_classifier.run(self.y_conv, feed_dict={self.Xmn: Images, self.keep_prob: 1.0})
        Curr_Labels = np.argmax(Curr_Preds, 1)
        return Curr_Preds, Curr_Labels

    def Evaluate_Labels_v2(self, Images):
        Curr_Preds = self.sess_classifier.run(self.y_conv, feed_dict={self.Xmn: Images, self.keep_prob: 1.0})
        Curr_Labels = np.argmax(Curr_Preds, 1)
        return Curr_Preds, Curr_Labels
