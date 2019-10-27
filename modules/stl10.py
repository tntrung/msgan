import os, sys
import numpy as np
from skimage.transform import resize

from modules.fiutils import *
from modules.imutils import imwrite

def create_stl10(source = 'unlabeled_X.bin', outdir = 'slt10'):
    '''
    Generate SLT-10 images from matlab files.
    '''
    with open(source, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        images = images.astype(float) / 255.0
        
        if not os.path.exists(outdir):
            os.mkdirs(outdir)

        nb_imgs = np.shape(images)[0]
        for ii in range(nb_imgs):
            img = resize(images[ii,:,:,:], [48, 48])
            imwrite(img, os.path.join(outdir, 'image_%06d.png' %(ii)))
                    
def load_stl10(data_dir):
    
    flag = os.path.exists(os.path.join(data_dir))
    
    if flag == False:
        # download and unzip cifar data
        filepath = download('http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', data_dir);
        #filepath = data_dir + '/stl10_binary.tar.gz'
        decompress(filepath, data_dir);
        # stl10_binary.tar.gz is extracted into 'stl10_binary'
        # copy the files out and delete the folder
        decom_dir = os.path.join(data_dir, 'stl10_binary')
        data_bin  = os.path.join(decom_dir, 'unlabeled_X.bin')
        create_stl10(data_bin,data_dir)
        print('[stl10.py - load_stl10] removing %s' % (decom_dir))
        remove_dir(decom_dir)
        os.remove(os.path.join(data_dir, 'stl10_binary.tar.gz'))
    
    
    

