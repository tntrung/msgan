# File Utils

import os, sys
import tarfile
import zipfile
import shutil
import requests

py_version = sys.version_info[0]

def extract_filename_of(filepath):
    filename = os.path.basename(filepath)
    return filename

def mkdirs(new_dir):
    if not os.path.exists(new_dir):
        print('[fiutils.py -- mkdirs] creating dir: %s' % (new_dir))
        os.makedirs(new_dir)

def copy_all_files(src_dir, dst_dir):
    src_files = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst_dir)

def remove_dir(src_dir):
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)
        
'''
Downloading data and save into specific folders
'''
def download(filelink, save_dir):
    filename = extract_filename_of(filelink)
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        mkdirs(save_dir)
        print('[fiutils.py -- download] downloading %s' % (filelink))
        if py_version < 3:
            import urllib
            urllib.urlretrieve(filelink, filepath)
        else:
            import urllib
            urllib.request.urlretrieve(filelink, filepath)
        print('[fiutils.py -- download]: saved at %s' % (filepath))
    return filepath


'''
Downloading data from Google Drive
'''

def download_file_from_google_drive(id, destination, filename):
    
    URL = "https://docs.google.com/uc?export=download"
    
    print('[fiutils.py -- download_file_from_google_drive] saving at: %s' % (destination))
    
    if not os.path.exists(destination):
       mkdirs(destination)
       
    print('[fiutils.py -- download_file_from_google_drive] downloading %s' % (URL + '&id=%s' %(id)))
        
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    filepath = destination + '/' + filename

    save_response_content(response, filepath)    
    
    return filepath

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def decompress(filepath, save_dir):
    print('[fiutils.py -- decompress] decompressing file: %s into %s' % (filepath, save_dir))
    if filepath.endswith("tar.gz"):
        tar = tarfile.open(filepath, "r:gz")
        tar.extractall(save_dir)
        tar.close()
    elif filepath.endswith("tar"):
        tar = tarfile.open(filepath, "r:")
        tar.extractall(save_dir)
        tar.close()
    elif filepath.endswith("zip"):
        with zipfile.ZipFile(filepath, 'r') as zipObj:
            zipObj.extractall(save_dir)
    else:
        print('[fiutils.py -- decompress] cannot decompress file: %s' % (filepath))
        exit()
