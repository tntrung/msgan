'''
************************************************************************
Implementation of SS/MS-DistGAN model by the authors of the paper:
"Self-supervised GAN: Analysis and Improvement with Multi-class Minimax 
Game", NeurIPS 2019.
************************************************************************
'''

import os
import numpy as np
import tensorflow as tf
import time

from modules.imutils import *
from modules.mdutils import *
from modules.vsutils import *
from modules.net_dcgan  import  *
from modules.net_sngan  import  *
from modules.net_resnet import  *

from support.mnist_classifier import classify

class MSGAN(object):

    def __init__(self, model='gan',     \
                 is_train = 0,          \
                 ss_task  = 2,          \
                 lambda_p = 1.0,        \
                 lambda_d = 0.5,        \
                 lambda_g = 0.1,        \
                 lr=2e-4, beta1 = 0.5, beta2 = 0.9,     \
                 noise_dim = 128,                       \
                 nnet_type='resnet',                    \
                 loss_type='hinge',                     \
                 df_dim = 64, gf_dim = 64, ef_dim = 64, \
                 dataset = None, batch_size = 64,       \
                 nb_test_real = 10000,                  \
                 nb_test_fake = 5000,                   \
                 n_steps = 300000,                      \
                 decay_step = 10000, decay_rate = 1.0,  \
                 log_interval=10,                       \
                 out_dir = './output/',                 \
                 verbose = True):
        """
        Initializing MS-Dist-GAN model
        """
        self.verbose      = verbose
        
        print('\n[msgan.py -- __init__] Intializing ... ')
        # dataset
        self.dataset   = dataset
        self.db_name   = self.dataset.db_name()
        print('[msgan.py -- __init__] db_name = %s' % (self.db_name))

        # training parameters
        self.model      = model
        self.is_train   = is_train
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.n_steps    = n_steps
        self.batch_size = self.dataset.mb_size()
        
        if self.verbose == True:
            print('[msgan.py -- __init__] model = %s, lr = %s, beta1 = %f, beta2 = %f, decay_step = %d, decay_rate = %f' % (self.model, self.lr, self.beta1, self.beta2, self.decay_step, self.decay_rate))
            print('[msgan.py -- __init__] n_steps = %d, batch_size = %d' % (self.n_steps, self.batch_size))

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim    = ef_dim
        self.gf_dim    = gf_dim
        self.df_dim    = df_dim
        
        if self.verbose == True:
            print('[msgan.py -- __init__] nnet_type = %s, loss_type = %s' % (self.nnet_type, self.loss_type))
            print('[msgan.py -- __init__] ef_dim = %d, gf_dim = %d, df_dim = %d' % (self.ef_dim, self.gf_dim, self.df_dim))
        
        # new constraints
        self.ss_task   = ss_task
        self.lambda_d  = lambda_d
        self.lambda_g  = lambda_g
                
        if self.verbose == True:
            print('[msgan.py -- __init__] ss_task = %d, lambda_d = %f, lambda_g = %f' % (self.ss_task, self.lambda_d, self.lambda_g))

        # dimensions
        self.data_dim   = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim  = noise_dim
        
        if self.verbose == True:
            print('[msgan.py -- __init__] data_dim = %d, noise_dim = %d' % (self.data_dim, self.noise_dim))
            print('[msgan.py -- __init__] data_shape = {}'.format(self.data_shape))

        # pamraeters
        self.lambda_p  = lambda_p
                        
        if self.verbose == True:
            print('[msgan.py -- __init__] lambda_p = %f' % (self.lambda_p))
        
        self.nb_test_real = nb_test_real
        self.nb_test_fake = nb_test_fake
        
        if self.verbose == True:
            print('[msgan.py -- __init__] FID: nb_test_real = %d, nb_test_fake = %d' % ( self.nb_test_real, self.nb_test_fake ))

        # others
        self.out_dir      = out_dir
        self.ckpt_dir     = out_dir + '/model/'
        self.log_file     = out_dir + '.txt'
        self.log_interval = log_interval
                
        if self.verbose == True:
            print('[msgan.py -- __init__] out_dir = {}'.format(self.out_dir))
            print('[msgan.py -- __init__] ckpt_dir = {}'.format(self.ckpt_dir))
            print('[msgan.py -- __init__] log_interval = {}'.format(self.log_interval))
            print('[msgan.py -- __init__] verbose = {}'.format(self.verbose))
        
        print('[msgan.py -- __init__] Done.')

        self.create_model()
        
        if self.db_name in ['mnist'] and self.noise_dim == 2:
            # Train classifier for MNIST to visualize latent space
            self.Classifier = classify()
            self.Classifier.TrainwithoutSave(dataset.db_source())

    def sample_z(self, N):
        return np.random.uniform(-1.0,1.0,size=[N, self.noise_dim])

    def create_discriminator(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return discriminator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
            return discriminator_dcgan_stacked_mnist              
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return discriminator_dcgan_celeba
        elif self.nnet_type == 'dcgan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return discriminator_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return discriminator_sngan_cifar            
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return discriminator_sngan_stl10
        elif self.nnet_type == 'resnet' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return discriminator_resnet_cifar   
        elif self.nnet_type == 'resnet' and self.db_name == 'stl10':          
            return discriminator_resnet_stl10               
        else:
            print('[msgan.py -- create_discriminator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));
            
    def create_generator(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return generator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
            return generator_dcgan_stacked_mnist  
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return generator_dcgan_celeba    
        elif self.nnet_type == 'dcgan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return generator_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return generator_sngan_cifar            
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return generator_sngan_stl10
        elif self.nnet_type == 'resnet' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return generator_resnet_cifar
        elif self.nnet_type == 'resnet' and self.db_name == 'stl10':
            return generator_resnet_stl10               
        else:
            print('[msgan.py -- create_generator] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));
            
    def create_encoder(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return encoder_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'mnist-1k':
            return encoder_dcgan_stacked_mnist  
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return encoder_dcgan_celeba
        elif self.nnet_type == 'dcgan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return encoder_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return encoder_dcgan_cifar           
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return encoder_sngan_stl10
        elif self.nnet_type == 'resnet' and self.db_name in ['cifar10','cifar100', 'imagenet_32']:
            return encoder_resnet_cifar  
        elif self.nnet_type == 'resnet' and self.db_name == 'stl10':
            return encoder_resnet_stl10              
        else:
            print('[msgan.py -- create_encoder] The dataset %s are not supported by the network %s' %(self.db_name, self.nnet_type));           

    def create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)    

    def create_model(self):

        self.X   = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
        self.z   = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
        self.zn  = tf.placeholder(tf.float32, shape=[None, self.noise_dim]) # to generate flexible number of images
        
        self.iteration = tf.placeholder(tf.int32, shape=None)

        # argument real samples for SS and MS task
        if self.ss_task  == 1:  # SS task
           self.Xarg, self.larg, self.ridx = tf_argument_image_rotation(self.X, self.data_shape)
        elif self.ss_task == 2: # MS task
           self.Xarg, self.larg, self.ridx = tf_argument_image_rotation_plus_fake(self.X, self.data_shape)
           
        # create generator
        with tf.variable_scope('generator'):
            self.G    = self.create_generator()
            self.X_f  = self.G(self.z,   self.data_shape, dim = self.gf_dim, reuse=False)   # to generate fake samples
            self.X_fn = self.G(self.zn,  self.data_shape, dim = self.gf_dim, reuse=True)    # to generate flexible number of fake images
                        
            # argument fake samples
            if self.ss_task == 1: # SS task
               self.Xarg_f, self.larg_f, _ = tf_argument_image_rotation(self.X_f, self.data_shape, self.ridx)
            elif self.ss_task == 2: # MS task
               self.Xarg_f, self.larg_f, _ = tf_argument_image_rotation_plus_fake(self.X_f,  self.data_shape, self.ridx)
        
        # MS task: argument real + fake samples
        if self.ss_task == 2:
            self.Xarg_mix, self.larg_mix, _ = tf_argument_image_rotation_and_fake_mix(self.X, self.X_f, self.data_shape)
        
        # create discriminator
        with tf.variable_scope('discriminator'):
            self.D   = self.create_discriminator()
            # D loss for SS/MS tasks
            if self.ss_task == 1 or self.ss_task == 2:
                self.d_real_sigmoid,  self.d_real_logit,  self.f_real,  _  = self.D(self.X,   self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=False)
                self.d_fake_sigmoid,  self.d_fake_logit,  self.f_fake,  _  = self.D(self.X_f, self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=True)
            else: # original D loss
                self.d_real_sigmoid,  self.d_real_logit,  self.f_real  = self.D(self.X,   self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=False)
                self.d_fake_sigmoid,  self.d_fake_logit,  self.f_fake  = self.D(self.X_f, self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=True)
                                
            # compute gradient penalty for discriminator loss
            epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0],1], minval=0., maxval=1.)
            interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
            if self.ss_task == 1 or self.ss_task == 2:
                _,d_inter,_, _ = self.D(interpolation, self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=True)
            else:
                _,d_inter,_ = self.D(interpolation, self.data_shape, dim = self.df_dim, ss_task = self.ss_task, reuse=True)
            gradients = tf.gradients([d_inter], [interpolation])[0]
            slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
            self.penalty = tf.reduce_mean((slopes - 1) ** 2)

            # compute SS loss
            if self.ss_task == 1:
                
                # predict real/fake classes of argumented samples with classifier
                _,  _,  _, self.real_cls = self.D(self.Xarg,   self.data_shape,   dim = self.df_dim, ss_task = self.ss_task, reuse=True)
                _,  _,  _, self.fake_cls = self.D(self.Xarg_f, self.data_shape,   dim = self.df_dim, ss_task = self.ss_task, reuse=True)

                # losses with softmax for D and G
                self.d_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
                self.g_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))
                
            # compute MS loss    
            elif self.ss_task == 2:
				
                # predict real/fake classes
                _,  _,  _, self.real_cls = self.D(self.Xarg,    self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                _,  _,  _, self.fake_cls = self.D(self.Xarg_f,  self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                _,  _,  _, self.mixe_cls = self.D(self.Xarg_mix,self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                
                # losses with softmax for D and G
                self.d_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.mixe_cls, labels=self.larg_mix))
                self.g_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))

        # Decay the weight of reconstruction for ResNet architecture
        t = tf.cast(self.iteration, tf.float32)/self.n_steps
        # mu = 0 if t <= N/2, mu in [0,0.05] 
        # if N/2 < t and t < 3N/2 and mu = 0.05 if t > 3N/2
        self.mu = tf.maximum(tf.minimum((t*0.1-0.05)*2, 0.05),0.0)
        w_real  = 0.95 + self.mu
        w_recon = 0.05 - self.mu
        w_fake  = 1.0

        # Discriminator loss with log function
        if self.loss_type == 'log':
            # Discriminator Loss
            self.d_real   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit, labels=tf.ones_like(self.d_real_sigmoid)))
            self.d_fake   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.zeros_like(self.d_fake_sigmoid)))
            self.d_cost_gan  = self.d_real + self.d_fake + self.lambda_p * self.penalty
                    
            # Generator loss
            self.g_cost_gan  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.ones_like(self.d_fake_sigmoid)))
            
        # Discriminator loss with hinge loss function
        elif self.loss_type == 'hinge':
            if self.nnet_type == 'dcgan':
                self.d_cost_gan  = -(tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_logit))     +  \
                                    tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_logit)) + self.lambda_p * self.penalty)
                self.g_cost_gan = - tf.reduce_mean(tf.minimum(0.,-1 + self.d_fake_logit))
            else:
                self.d_cost_gan = -(tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_sigmoid))   +  \
                                    tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_sigmoid)) + self.lambda_p * self.penalty)            
                self.g_cost_gan = - tf.reduce_mean(tf.minimum(0.,-1 + self.d_fake_sigmoid))

        # Combine GAN task and SS task
        if self.ss_task > 0:
           self.d_cost = self.d_cost_gan + self.lambda_d * self.d_acc
           self.g_cost = self.g_cost_gan + self.lambda_g * self.g_acc
        else:
           self.d_cost = self.d_cost_gan    
           self.g_cost = self.g_cost_gan

        # Create optimizers        
        if self.nnet_type == 'resnet':

            self.vars_g = [var for var in tf.trainable_variables() if 'generator' in var.name]
            self.vars_d = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            
            print('[msgan.py -- create_model] ********** parameters of Generator **********')
            print(self.vars_g) 
            print('[msgan.py -- create_model] ********** parameters of Discriminator **********')
            print(self.vars_d)
            
            self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
                        
            if self.is_train == 1:
                
                self.decay_rate = tf.maximum(0., tf.minimum(1.-(tf.cast(self.iteration, tf.float32)/self.n_steps),0.5))
                
                self.opt_gen = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1, beta2=self.beta2)
                self.opt_dis = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1, beta2=self.beta2)
                
                self.gen_gv  = self.opt_gen.compute_gradients(self.g_cost, var_list=self.vars_g)
                self.dis_gv  = self.opt_dis.compute_gradients(self.d_cost, var_list=self.vars_d)
                                
                self.opt_g  = self.opt_gen.apply_gradients(self.gen_gv)
                self.opt_d  = self.opt_dis.apply_gradients(self.dis_gv)
            
        else:
            
            # Create optimizers
            self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                                    
            print('[msgan.py -- create_model] ********** parameters of Generator **********')
            print(self.vars_g)
            print('[msgan.py -- create_model] ********** parameters of Discriminator **********')
            print(self.vars_d)
            
            self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            
            if self.is_train == 1:

                # Setup for weight decay
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

                self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
                self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
        
        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Training the model
        """
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        
        fid = open(self.log_file,"w")
        
        saver = tf.train.Saver(var_list = self.vars_g_save + self.vars_d_save, max_to_keep=1)
       
        with tf.Session(config=run_config) as sess:
            
            start = time.time()
            sess.run(self.init)
                       
            for step in range(self.n_steps + 1):

                # train discriminator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_d],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                # train generator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_g],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                # compute losses to print
                if self.ss_task > 0:
                    loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc = \
                    sess.run([self.d_cost, self.d_cost_gan, self.d_acc, self.g_cost, self.g_cost_gan, self.g_acc],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                else:
                    loss_d, loss_g = sess.run([self.d_cost, self.g_cost], feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                if step % self.log_interval == 0:
                    if self.verbose:
                       elapsed = int(time.time() - start)
                       if self.ss_task > 0:
                          output_str = '[msgan.py -- train] step: {:4d}, D loss: {:8.4f}, D loss (gan): {:8.4f}, D loss (acc): {:8.4f} G loss: {:8.4f}, G loss (gan): {:8.4f}, G loss (acc): {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc, elapsed)
                       else:
                          output_str = '[msgan.py -- train] step: {:4d}, D loss: {:8.4f}, D loss (gan): {:8.4f}, D loss (acc): {:8.4f} G loss: {:8.4f}, G loss (gan): {:8.4f}, G loss (acc): {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_d, 0, loss_g, loss_g, 0, elapsed)
                       print(output_str)
                       fid.write(str(output_str)+'\n')
                       fid.flush()

                if step % (self.log_interval*1000) == 0:
                    # save real images
                    im_save_path = os.path.join(self.out_dir,'image_%d_real.jpg' % (step))
                    imsave_batch(mb_X, self.data_shape, im_save_path)
                    
                    # save generated images
                    im_save_path = os.path.join(self.out_dir,'image_%d_fake.jpg' % (step))
                    mb_X_f = sess.run(self.X_f,feed_dict={self.z: mb_z})
                    imsave_batch(mb_X_f, self.data_shape, im_save_path)
                    
                    if self.ss_task > 0:
                        # save argumented images
                        Xarg = sess.run(self.Xarg,feed_dict={self.X: mb_X, self.z: mb_z})                    
                        im_save_path = os.path.join(self.out_dir,'image_%d_real_argu.jpg' % (step))
                        imsave_batch(Xarg, self.data_shape, im_save_path)
                        
                        if self.ss_task == 2:
                            # save mix argumented images
                            Xarg_mix = sess.run(self.Xarg_mix,feed_dict={self.X: mb_X, self.z: mb_z})
                            im_save_path = os.path.join(self.out_dir,'image_%d_mixe_argu.jpg' % (step))
                            imsave_batch(Xarg_mix, self.data_shape, im_save_path)
                                                
                if step % (self.log_interval*1000) == 0:
                                     
                    if step == 0:
                        real_dir = self.out_dir + '/real/'
                        if not os.path.exists(real_dir):
                            os.makedirs(real_dir)
                            
                    fake_dir = self.out_dir + '/fake_%d/'%(step)
                    if not os.path.exists(fake_dir):
                        os.makedirs(fake_dir)
                        
                    #generate real samples to compute FID
                    if step == 0:
                        for v in range(self.nb_test_real // self.batch_size + 1):
                            mb_X = self.dataset.next_batch()
                            im_real_save = np.reshape(mb_X,(-1, self.data_shape[0], self.data_shape[1],self.data_shape[2]))
                            for ii in range(np.shape(mb_X)[0]):
                                real_path = real_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_real]))
                                imwrite(im_real_save[ii,:,:,:], real_path)
                    #generate fake samples to compute FID            
                    elif step > 0:
                        for v in range(self.nb_test_fake // self.batch_size + 1):
                            mb_z = self.sample_z(np.shape(mb_X)[0])
                            im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                            im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                            for ii in range(np.shape(mb_z)[0]):
                                fake_path = fake_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                                imwrite(im_fake_save[ii,:,:,:], fake_path)

                if step > 0 and step % int(self.n_steps/2) == 0:
                    if not os.path.exists(self.ckpt_dir +'%d/'%(step)):
                        os.makedirs(self.ckpt_dir +'%d/'%(step))
                    save_path = saver.save(sess, '%s%d/epoch_%d.ckpt' % (self.ckpt_dir, step,step))
                    print('[msgan.py -- train] the trained model is saved at: % s' % save_path)
