import os
import sys

from modules.fiutils import *

from skimage.transform import resize

# processing for celeba dataset
def preprocess(img):
    crop_size = 108
    re_size   = 64
    top_left  = [(218 - crop_size)//2, (178 - crop_size)//2]
    img       = img[top_left[0]:top_left[0]+crop_size, top_left[1]:top_left[1]+crop_size, :]
    img       = resize(img, [re_size, re_size])
    return img

# load cifar10        
def load_celeba(data_dir):
    
    # check and download data first
    flag = os.path.exists(os.path.join(data_dir))
    if flag == False:
        # download and unzip cifar data
        #filepath = download('https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0', data_dir);
        filepath = download_file_from_google_drive('0B7EVK8r0v71pZjFTYXZWM3FlRnM', data_dir, 'celeba.zip');
        #filepath = data_dir + '/celeba.zip'
        decompress(filepath, data_dir);
        # celeba.zip is extracted into 'img_align_celeba'
        # copy the files out and delete the folder
        decom_dir = os.path.join(data_dir, 'img_align_celeba')
        copy_all_files(decom_dir, data_dir)
        print('[celeba.py - load_celeba] removing %s' % (decom_dir))
        remove_dir(decom_dir)
        os.remove(os.path.join(data_dir, 'celeba.zip'))
