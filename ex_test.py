from glob import glob
import random
import os

DIR_PATH = os.getcwd()
TRAIN_PATH = DIR_PATH + '/train/'
VAL_PATH = DIR_PATH + '/val/'

# Function to rename files
def rename_files(path):
    os.chdir(path)
    files = glob('*.jpg')
    for file in files:
        new_name = file.replace('_1', '_1.0')
        # new_name = file.replace('_0', '_0.0')
        os.rename(file, new_name)

# Rename files in 'nothreats' folder
rename_files(TRAIN_PATH + 'nothreats/')

# Rename files in 'threats' folder
rename_files(TRAIN_PATH + 'threats/')

# Rename files in 'nothreats' folder
rename_files(VAL_PATH + 'nothreats/')

# Rename files in 'threats' folder
rename_files(VAL_PATH + 'threats/')
