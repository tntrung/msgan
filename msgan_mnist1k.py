'''
************************************************************************
Implementation of SS/MS-DistGAN model by the authors of the paper:
"Self-supervised GAN: Analysis and Improvement with Multi-class Minimax 
Game", NeurIPS 2019.
************************************************************************
'''

import os, sys
import numpy as np
from   msdistgan import MSDistGAN
from   modules.dataset import Dataset
from   modules.eval import generate_fake_samples, compute_mode_kl

if __name__ == '__main__':
    
    '''
    ********************************************************************
    * Database and outputs
    ********************************************************************
    '''
    db_name       = 'mnist-1k'
    out_dir       = './output/'
    data_source   = './data/mnist-1k/'
    
               
    is_train      = 0 # 0 (train model and compute FID after training)
                      # 1 (compute FID of pre-trained model)
    
    '''
    The architectures of Unrolled-GAN with N feature maps.
    @K = 4: N/4 architecture
    @K = 2: N/2 architecture
    @K = 1: N/1 architecture (full)              
    '''                  
    K = 2
    
    '''
    ********************************************************************
    * Network architectures and objective losses
    ********************************************************************
    '''
    '''
    To apply SS and SS into Dist-GAN model: Refer to Dist-GAN paper
    for more details: https://arxiv.org/abs/1803.08887
    '''    
    model     = 'gan'
    
    '''
    network architecture supports: 'dcgan'
    '''
    nnet_type = 'dcgan'
    '''
    objective loss type supports: 'log'
    '''
    loss_type = 'log'
    '''
    0: no use ss/ms task (original DistGAN)
    1: the original self-supervised task (SS task)
    2: the multi-class minimax self-supervised task (MS task)
    To select suggested @lambda_d and @lambda_g for SS or MS tasks
    '''
    ss_task = 2
           
      
    '''
    ********************************************************************
    * Training, network architectures and model parameters
    ********************************************************************
    '''    
    n_steps   = 200000    # number of iterations
    noise_dim = 256       # the noise dimension
    
    '''
    The dimension of feature size for original dist-gan model. 
    If you're using our pre-defined datasets, keep it!
    Otherwise, adapt to new feature_dim for your new dataset.
    '''
    if K == 4:
       feature_dim  = 128  #feture dim (N/4)
    elif K == 2:
       feature_dim  = 256  #feture dim (N/2)
    elif K == 1:
       feature_dim  = 512  #feture dim (N/1)
    
    '''
    The unit dimensions for network architectures.
    @df_dim: feature map unit for discriminator.
    @gf_dim: feature map unit for generator.
    @ef_dim: feature map unit for encoder.
    @lr: learning rate
    @beta1, beta2 parameters for Adam optimizer
    '''
    
    if nnet_type == 'dcgan':
        
        #1: N/1 architecture, 2: N/2 architecture, 4: N/4 architecture
        df_dim = int(8 * 1.0 / K) 
        gf_dim = 8
        ef_dim = 8
        lr     = 2e-4
        beta1  = 0.5
        beta2  = 0.9
        
    '''
    Selected pamraeters for SS and MS tasks
    @lambda_d: SS/MS for discriminator learning
    @lambda_g: SS/MS for generator learning
    '''
        
    if nnet_type == 'dcgan':
        if ss_task == 1:
           if K == 4:
               lambda_d  = 0.5   
               lambda_g  = 0.0             
           elif K == 2:
               lambda_d  = 1.0   
               lambda_g  = 0.0 
           elif K == 1:
               # haven't verified yet
               lambda_d  = 1.0   
               lambda_g  = 0.0 
        elif ss_task == 2:
           if K == 4:   
               lambda_d  = 0.5   
               lambda_g  = 0.2 
           elif K == 2:
               lambda_d  = 1.0   
               lambda_g  = 1.0
           elif K == 1:
               # haven't verified yet
               lambda_d  = 1.0   
               lambda_g  = 1.0              
        else:
           lambda_d = 0.0
           lambda_g = 0.0  
                  
    batch_size = 64 # bach size for each iteration
    
    lambda_p  = 0.5  # gradient-penalty term

    '''
    ********************************************************************
    * Training and testing
    ********************************************************************
    '''
    ext_name = 'K_%d_sstask_%d_ld_%.02f_lg_%.02f_batch_%d_niters_%d' \
                   % (K, ss_task, lambda_d, lambda_g , batch_size, n_steps)
        
    #output dir
    out_dir = os.path.join(out_dir, db_name + '_' + model + '_'     \
                                                  + nnet_type + '_' \
                                                  + loss_type + '_' \
                                                  + ext_name, db_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    msgan = MSGAN(model=model,               \
                              is_train  = is_train,  \
                              ss_task   = ss_task,   \
                              loss_type = loss_type, \
                              lambda_p  = lambda_p,  \
                              lambda_d  = lambda_d,  \
                              lambda_g  = lambda_g,  \
                              noise_dim = noise_dim, \
                              lr    = lr,            \
                              beta1 = beta1,         \
                              beta2 = beta2,         \
                              nnet_type = nnet_type, \
                              df_dim = df_dim,       \
                              gf_dim = gf_dim,       \
                              ef_dim = ef_dim,       \
                              dataset=dataset,       \
                              n_steps = n_steps,     \
                              out_dir=out_dir)
    if is_train == 0:
        msgan.train()
    elif is_train == 1:
        '''
        evaluating the mode and kl of mnist-1k
        '''
        fake_dir = out_dir + '/fake_samples/'
        print('[msdistgan_mnist1k.py -- main] output_dir: %s' %(fake_dir))
        generate_fake_samples(msgan, fake_dir, n_steps = n_steps)
        compute_mode_kl(fake_dir, is_train = 0)
