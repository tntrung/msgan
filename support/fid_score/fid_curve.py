#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

def fid_example():
    # Paths
    re_est_gth = False
  
    dbname     = 'cifar10'
    input_dir  = '../../gan/output/'
    start      = 10000
    niters     = 300000
    step       = 10000

    if dbname == 'cifar10':
        model = 'cifar10_wgangp_dcgan_wdis_lp_10_300000'
    elif dbname == 'stl10':
        model = 'stl10_distgan_resnet_hinge_gngan_0_ssgan_3_ld_1.0_lg_0.010_300000'
        
    mu_gth_file    = 'mu_gth_' + dbname + '_10k.npy'
    sigma_gth_file = 'sigma_gth_' + dbname + '_5k.npy'

    """
    # loads all images into memory (this might require a lot of RAM!)
    gth_list = glob.glob(os.path.join(gth_path, '*.jpg'))
    gen_list = glob.glob(os.path.join(gen_path, '*.jpg'))
    gth_images = np.array([imread(str(fn)).astype(np.float32) for fn in gth_list])
    gen_images = np.array([imread(str(fn)).astype(np.float32) for fn in gen_list])
    """
    """
    # load precalculated training set statistics
    f = np.load(path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    """
    print('FID ESTIMATE')

    import os
    import os.path

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    inception_path = fid.check_or_download_inception('/tmp') # download inception network

    logfile = os.path.join(input_dir, model, dbname + '_' + model + '_fid_%d_%d.txt'%(start,niters))
    print(logfile)
    fid_log = open(logfile, 'w')
    
    if os.path.isfile(mu_gth_file) and os.path.isfile(sigma_gth_file) and re_est_gth:
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        mu_gth = np.load(mu_gth_file)
        sigma_gth = np.load(sigma_gth_file)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(start,niters+1,step):
                gen_path = os.path.join(input_dir, model, dbname, 'fake_%d'%i) # set path to some generated images
                print('[%s]'%(gen_path))
                mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                strout = "step: %d - FID: %s" % (i, fid_value)
                print(strout)
                fid_log.write(strout + '\n')
                fid_log.flush()
                
    else:
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gth_path = os.path.join(input_dir, model, dbname, 'real') # set path to some ground truth images
            mu_gth, sigma_gth = fid._handle_path(gth_path, sess)
            for i in range(start,niters+1,step):
                gen_path = os.path.join(input_dir, model, dbname, 'fake_%d'%i) # set path to some generated images
                print('[%s]'%(gen_path))
                mu_gen, sigma_gen = fid._handle_path(gen_path, sess)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gth, sigma_gth)
                strout = "step: %d - FID: %s" % (i, fid_value)
                print(strout)
                fid_log.write(strout + '\n')
                fid_log.flush()
                
        np.save(mu_gth_file, mu_gth)
        np.save(sigma_gth_file, sigma_gth)

    return fid_value

if __name__ == '__main__':
    fid_example()
