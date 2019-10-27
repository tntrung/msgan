'''
************************************************************************
Implementation of SS/MS-DistGAN model by the authors of the paper:
"Self-supervised GAN: Analysis and Improvement with Multi-class Minimax 
Game", NeurIPS 2019.
************************************************************************
'''

import os, sys
import numpy as np
import argparse
from   msdistgan import MSDistGAN
from   modules.dataset import Dataset
from   modules.eval import compute_fid_score

if __name__ == '__main__':

    '''
    ********************************************************************
    * Command-line arguments
    ********************************************************************
    ''' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',       type=int,   default=0,                  help='The ID of the specified GPU')
    parser.add_argument('--is_train',     type=int,   default=1,                  help='train: 1, test: 0')
    parser.add_argument('--db_name',      type=str,   default='cifar10',          help='Database options: cifar10 or cifar100')
    parser.add_argument('--nnet_type',    type=str,   default='resnet',           help='Network architectures: dcgan, sngan and resnet')
    parser.add_argument('--loss_type',    type=str,   default='hinge',            help='Loss types: log or hinge')
    parser.add_argument('--out_dir',      type=str,   default='./output/',        help='The output path of the program')
    parser.add_argument('--data_source',  type=str,   default='./data/cifar10/',  help='The dataset path to store (if data is unavailable, it is downloaded automatically)')
    parser.add_argument('--ss_task',      type=int,   default=2,                  help='0: no SS, 1: SS task, 2: MS task (self-supervised task)')
    parser.add_argument('--n_steps',      type=int,   default=300000,             help='The number of training iterations')
    parser.add_argument('--noise_dim',    type=int,   default=128,                help='The dimension of latent noise')
    parser.add_argument('--batch_size',   type=int,   default=64,                 help='Mini-batch size')
    parser.add_argument('--nb_test_real', type=int,   default=10000,              help='Number of real samples to compute FID')
    parser.add_argument('--nb_test_fake', type=int,   default=5000,               help='Number of fake samples to compute FID')
    
    opt = parser.parse_args()    
    '''
    ********************************************************************
    * Database and outputs
    ********************************************************************
    '''
    db_name       = opt.db_name   # 'cifar10' or 'cifar100'
    out_dir       = opt.out_dir
    data_source   = opt.data_source
    
    if db_name not in ['cifar10', 'cifar100']:
       print('\n[msdisgan_cifar.py -- __main__] %s is not supported.' % (db_name))
       exit()
           
    is_train      = opt.is_train # 1 (train model and compute FID after training)
                                 # 0 (compute FID of pre-trained model)
    
    '''
    Number of real or generated samples to compute FID scores
    '''                  
    nb_test_real = opt.nb_test_real
    nb_test_fake = opt.nb_test_fake               
    
    '''
    ********************************************************************
    * Network architectures and objective losses
    ********************************************************************
    '''
    '''
    To apply SS and SS into Dist-GAN model: Refer to Dist-GAN paper
    for more details: https://arxiv.org/abs/1803.08887
    '''    
    model     = 'distgan'
    
    '''
    network architecture supports: 'dcgan', 'sngan', 'resnet' 
    '''
    nnet_type = opt.nnet_type
    '''
    objective loss type supports: 'log' or 'hinge'
    '''
    loss_type = opt.loss_type
    '''
    0: no use ss/ms task (original dist-gan)
    1: the original self-supervised task (SS task)
    2: the multi-class minimax self-supervised task (MS task)
    To select suggested @lambda_d and @lambda_g for SS or MS tasks
    '''
    ss_task = opt.ss_task
    
    '''
    Selected pamraeters for SS and MS tasks
    @lambda_d: SS/MS for discriminator learning
    @lambda_g: SS/MS for generator learning
    '''
    if nnet_type == 'dcgan':
        if ss_task == 1:
           lambda_d = 1.0  # 1.0 for SS
           lambda_g = 0.0  # 0.0 for SS
        elif ss_task == 2:
           lambda_d = 1.0  # 1.0 for MS
           lambda_g = 0.1  # 0.1 for MS
        else:
           lambda_d = 0.0  
           lambda_g = 0.0 
    elif nnet_type == 'sngan':
        if ss_task == 1:
           lambda_d = 1.0   # 1.0 for SS
           lambda_g = 0.0   # 0.0 for SS
        elif ss_task == 2:
           lambda_d  = 1.0  # 1.0 for MS
           lambda_g  = 0.01 # 0.0 for MS
        else:
           lambda_d = 0.0  
           lambda_g = 0.0         
    elif nnet_type == 'resnet': 
        if ss_task == 1:
           lambda_d = 0.5  # 0.5 for SS
           lambda_g = 0.0  # 0.0 for SS
        elif ss_task == 2:
           lambda_d = 0.5  # 0.5 for MS
           lambda_g = 0.1  # 0.1 for MS
        else:
           lambda_d = 0.0  
           lambda_g = 0.0       
    else:
       print('\n[msdisgan_cifar.py -- __main__] %s is not supported.' % (nnet_type))
       exit()		
      
    '''
    ********************************************************************
    * Training, network architectures and model parameters
    ********************************************************************
    '''    
    n_steps   = opt.n_steps    # the number of iterations
    noise_dim = opt.noise_dim  # the noise dimension
    
    '''
    The dimension of feature size for original dist-gan model. 
    If you're using our pre-defined datasets, keep it!
    dcgan: 2048
    sngan: 8192
    resnet 8192
    Otherwise, adapt to new feature_dim for your new dataset.
    '''
    if nnet_type == 'sngan' or nnet_type == 'resnet':
        feature_dim = 8192
    elif nnet_type == 'dcgan':
        feature_dim = 2048
    
    '''
    The unit dimensions for network architectures.
    @df_dim: feature map unit for discriminator.
    @gf_dim: feature map unit for generator.
    @ef_dim: feature map unit for encoder.
    @lr: learning rate
    @beta1, beta2 parameters for Adam optimizer
    '''
    if nnet_type == 'resnet':
        df_dim = 128
        gf_dim = 128
        ef_dim = 128
        lr     = 2e-4
        beta1  = 0.0
        beta2  = 0.9
    else:
        df_dim = 64
        gf_dim = 64
        ef_dim = 64
        lr     = 2e-4
        beta1  = 0.5
        beta2  = 0.9
    
    batch_size = opt.batch_size # bach size for each iteration
    
    '''
    The fixed parameters for original Dist-GAN model. Refer to Dist-GAN
    paper for more details: https://arxiv.org/abs/1803.08887
    '''        
    lambda_p  = 0.5 # gradient-penalty term
    lambda_r  = 1.0 # data-latent distance term
    lambda_w  = np.sqrt(noise_dim * 1.0/feature_dim)
    
    '''
    ********************************************************************
    * Training and testing
    ********************************************************************
    '''
    ext_name = 'sstask_%d_ld_%.02f_lg_%.02f_batch_%d_niters_%d' \
                   % (ss_task, lambda_d, lambda_g , batch_size, n_steps)
        
    #output dir
    model_dir = db_name + '_' + model + '_'     \
                                                  + nnet_type + '_' \
                                                  + loss_type + '_' \
                                                  + ext_name
                                                  
    base_dir   = os.path.join(out_dir, model_dir, db_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source,\
												  batch_size=batch_size)
    
    # setup gan model and train
    msdistgan = MSDistGAN(model=model,               \
                              is_train  = is_train,  \
                              ss_task   = ss_task,   \
                              loss_type = loss_type, \
                              lambda_p  = lambda_p,  \
                              lambda_r  = lambda_r,  \
                              lambda_w  = lambda_w,  \
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
                              out_dir=base_dir)
    if is_train == 1:
        msdistgan.train()
        # compute fid score
        compute_fid_score(dbname = db_name,       \
                          input_dir = out_dir,    \
                          model = model_dir,      \
                          nb_train = nb_test_real,\
                          nb_test  = nb_test_fake)
    elif is_train == 0:
        # compute fid score
        compute_fid_score(dbname = db_name,       \
                          input_dir = out_dir,    \
                          model = model_dir,      \
                          nb_train = nb_test_real,\
                          nb_test  = nb_test_fake)
