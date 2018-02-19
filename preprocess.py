# This script generates all the needed .jpg files for training and testing 
# in Kaggles Passenger Screening Algorithm Challenge. My approach involves cropping
# the aps slices in general regions where the body zone is likely to be
# and stitching these together in an image plate of front/back
# /front side angle/back side angle 
#
# All aps files must be in aps directory 
#
# To preprocess train .aps files for training stage, run with -train arg
#
# To preprocess test .aps files for testing, run with -test arg
#
# Big shout out to Brian Farrar. This script was largely adapted from his notebooks 
# found here:
# https://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation
# https://www.kaggle.com/jbfarrar/preprocessing-pipeline-and-convnet-trainer

from __future__ import print_function
from __future__ import division
from timeit import default_timer as timer
from glob import glob
from shutil import copyfile
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.misc
import PIL
from PIL import Image
from pathlib import Path

DIR_PATH = os.getcwd()
TRAIN_PATH = DIR_PATH + '/train/'
VAL_PATH = DIR_PATH + '/val/'
TEST_PATH = DIR_PATH + '/test/'
I_FOLDER = DIR_PATH + '/aps'
TRAIN_LABELS = DIR_PATH + '/stage1_labels.csv'
TEST_LABELS = DIR_PATH + '/stage1_sample_submission.csv'
 


# Divide the available space on an image into 16 sectors. In the [0] image these
# zones correspond to the TSA threat zones.  But on rotated images, the slice
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)
sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32) 

sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

# crop dimensions, upper left y, x, width, height
sector_crop_list = [[ 145,  65, 111, 111], 
                    [  60,  65, 111, 111], 
                    [ 120, 285, 111, 111], 
                    [ 60, 315, 111, 111],
                    [180, 160, 111, 111],
                    [240, 150, 111, 111],
                    [250, 250, 111, 111], 
                    [350,  170, 111, 111], 
                    [350, 205, 111, 111], 
                    [350, 280, 111, 111], 
                    [450, 120, 111, 111], 
                    [450, 290, 111, 111], 
                    [410,   0, 111, 111], 
                    [410, 200, 111, 111], 
                    [410,   0, 111, 111], 
                    [410, 200, 111, 111], 
                    [120, 120, 111, 111], 
                    [50, 118, 111, 111],  
                    [130, 315, 111, 111], 
                    [150, 122, 111, 111],
                    [90, 60, 111, 111],   
                    [60, 100, 111, 111],  
                    [200, 200, 111, 111], 
                    [250, 185, 111, 111],  
                    [250, 210, 111, 111],
                    [350, 240, 111, 111],
                    [350, 235, 111, 111],
                    [350, 190, 111, 111],  #index 27
                    [430, 170, 111, 111],
                    [430, 205, 111, 111],
                    [450, 140, 111, 111],
                    [430, 160, 111, 111],
                    [450, 240, 111, 111],     #index 32
                    [520, 100, 111, 111],
                    [520, 150, 111, 111], 
                    [520, 310, 111, 111],
                    [520, 300, 111, 111],
                    [510, 150, 111, 111],    #index 37
                    [510, 250, 111, 111], 
                    [510, 185, 111, 111],
                    [510, 120, 111, 111],    #index 40
                    [540, 110, 111, 111], 
                    [540, 300, 111, 111], 
                    [540, 275, 111, 111], 
                    [540, 180, 111, 111],     #index 44
                    [540, 265, 111, 111],
                    [540, 220, 111, 111],
                    [540, 150, 111, 111], 
                    [540, 290, 111, 111],
                    [240, 240, 111, 111],     #index 49 
                    [230, 200, 111, 111], 
                    [230, 200, 111, 111],
                    [230, 170, 111, 111],
                    [350, 215, 111, 111],
                    [350, 200, 111, 111],    #index 54
                    [60, 260, 111, 111]
                    
                   ]


zone_slice_list = [ [ # threat zone 1
                      sector01_pts, None, None, None,
                      None, None, None, None, 
                      None, sector03_pts, None, 
                      None, None, sector03_pts, None, sector01_pts ], 
    
                    [ # threat zone 2
                      sector02_pts, None, None, None, 
                      None, None, None, None, 
                      None, sector04_pts, None, None, 
                      None, sector02_pts, sector02_pts, None ],
    
                    [ # threat zone 3
                      None, sector03_pts, sector03_pts, sector03_pts, 
                      None, None, sector01_pts, None,
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 4
                     None, None, sector04_pts, sector04_pts, 
                      None, None, sector04_pts, None, 
                      sector02_pts, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 5
                      sector05_pts, sector05_pts, sector05_pts, None, 
                      None, None, None, None,
                      None, None, None, None, 
                      None, None, sector05_pts, None ],
    
                    [ # threat zone 6
                      sector06_pts, None, None, None, 
                      None, None, None, None, 
                      sector07_pts, sector07_pts, None, None, 
                      None, sector06_pts, None, None ],
    
                    [ # threat zone 7
                      sector07_pts, None, None, sector07_pts, 
                      None, sector07_pts, None, sector07_pts, 
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 8
                      None, None, None, None, 
                      None, None, None, None, 
                      None, None, sector10_pts, sector10_pts, 
                      sector08_pts, sector08_pts, None, None ],
    
                    [ # threat zone 9
                      sector09_pts, None, sector08_pts, sector08_pts, 
                      None, None, None, None,
                      sector09_pts, None, None, None, 
                      None, None, None, None],
    
                    [ # threat zone 10
                      sector10_pts, None, None, sector10_pts, 
                      sector10_pts, None, sector10_pts, None, 
                      None, None, None, None, 
                      None, None, None, None],
    
                    [ # threat zone 11
                      sector11_pts, None, None, None, 
                      None, None, sector12_pts, sector12_pts,
                      None, None, None, None, 
                      sector11_pts, None, None, None ],
    
                    [ # threat zone 12
                      None, None, None, None, 
                      sector12_pts, None, None, None, 
                      sector11_pts, None, sector11_pts, None, 
                      None, None, None, sector12_pts ],
    
                    [ # threat zone 13
                      sector13_pts,None, None, None, 
                      None, None, None, sector14_pts,
                      sector14_pts, None, None, None, 
                      sector13_pts, None, None, None],
    
                    [ # sector 14
                      sector14_pts, None, None, None, 
                      sector14_pts, None, sector13_pts, None, 
                      None, None, sector13_pts, None, 
                      None, None, None, None ],
    
                    [ # threat zone 15
                      sector15_pts, None, None, None, 
                      None, None, sector16_pts, None,
                      None, sector16_pts, None, sector15_pts, 
                      None, None, None, None ],
    
                    [ # threat zone 16
                      None, None, None, sector16_pts, 
                      None, sector16_pts, None, sector15_pts, 
                      None, None, None, None, 
                      None, None, None, sector16_pts ],
    
                    [ # threat zone 17
                      None, None, None, None, 
                      None, None, None, None,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, 
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_crop_list =  [ [ # threat zone 1
                      sector_crop_list[0], None, None, None,
                      None, None, None, None, 
                      None, sector_crop_list[2], None, None, 
                      None, sector_crop_list[16], None, sector_crop_list[0] ],
    
                    [ # threat zone 2
                      sector_crop_list[1], None, None, None, 
                      None, None, None, None, None, 
                      sector_crop_list[3],
                      None, None, None, 
                      sector_crop_list[17], 
                      sector_crop_list[1], None],
    
                    [ # threat zone 3
                      sector_crop_list[18], sector_crop_list[18], sector_crop_list[18], 
                      sector_crop_list[18], None, None, sector_crop_list[19], 
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], 
                      sector_crop_list[0], sector_crop_list[0], None, None, 
                      sector_crop_list[18], sector_crop_list[18] ],
               
                    [ # threat zone 4
                      None, None, sector_crop_list[3], 
                      sector_crop_list[55], None, None, sector_crop_list[21], 
                      None, sector_crop_list[20], None, 
                      None, None, None, None, 
                      None, None ],
                    
                    [ # threat zone 5
                      sector_crop_list[4], sector_crop_list[22], sector_crop_list[22], 
                      sector_crop_list[22], sector_crop_list[22], sector_crop_list[22], 
                      sector_crop_list[22], sector_crop_list[22],
                      None, None, None, None, None, None, sector_crop_list[4], None ],
                    
                    [ # threat zone 6
                      sector_crop_list[5], None, None, None, None, None, None, None, 
                      sector_crop_list[6], sector_crop_list[6], None, 
                      None, None, sector_crop_list[23], 
                      None, None ],
    
                    [ # threat zone 7
                      sector_crop_list[49], None, None, 
                      sector_crop_list[50], None, sector_crop_list[51], 
                      None, sector_crop_list[51], 
                      None, None, None, None, None, None, None, None ],
    
                    [ # threat zone 8
                      None, None, None, None, None, 
                      None, None, None, None, 
                      None, sector_crop_list[53], sector_crop_list[54], 
                      sector_crop_list[7], sector_crop_list[7], None, 
                      None],
    
                    [ # threat zone 9
                      sector_crop_list[8], sector_crop_list[8], sector_crop_list[7], 
                      sector_crop_list[7], sector_crop_list[7], None, None, None,
                      sector_crop_list[8], sector_crop_list[8], None, None, None, 
                      None, sector_crop_list[9], sector_crop_list[8] ],
    
                    [ # threat zone 10
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], 
                      sector_crop_list[25], sector_crop_list[26], sector_crop_list[7], 
                      sector_crop_list[27], None, None, None, None, None, None, None, 
                      None, sector_crop_list[9] ],
    
                    [ # threat zone 11
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], 
                      sector_crop_list[10], None, None, sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], None, sector_crop_list[28], 
                      sector_crop_list[28], sector_crop_list[28], sector_crop_list[28] ],
    
                    [ # threat zone 12
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[29], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[30], 
                      sector_crop_list[11], sector_crop_list[31], None, None, 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[32] ],
    
                    [ # threat zone 13
                      sector_crop_list[33], sector_crop_list[34], sector_crop_list[12], 
                      sector_crop_list[12], None, None, sector_crop_list[35], 
                      sector_crop_list[35], sector_crop_list[36], sector_crop_list[13], 
                      sector_crop_list[13], None, sector_crop_list[37], 
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],
    
                    [ # sector 14
                      sector_crop_list[38], sector_crop_list[38], sector_crop_list[38], 
                      sector_crop_list[38], sector_crop_list[38], None, 
                      sector_crop_list[39], sector_crop_list[13], sector_crop_list[12], 
                      sector_crop_list[12], sector_crop_list[40], None, None, None, 
                      None, None ],
    
                    [ # threat zone 15
                      sector_crop_list[41], sector_crop_list[14], sector_crop_list[14], 
                      sector_crop_list[14], None, None, sector_crop_list[42], 
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[43], 
                      None, sector_crop_list[44], sector_crop_list[14], None, 
                      sector_crop_list[14], sector_crop_list[14] ],
    
                    [ # threat zone 16
                      sector_crop_list[45], sector_crop_list[15], sector_crop_list[15], 
                      sector_crop_list[45], sector_crop_list[15], sector_crop_list[46], 
                      sector_crop_list[14], sector_crop_list[47], sector_crop_list[14], 
                      sector_crop_list[14], sector_crop_list[14], None, None, None, 
                      sector_crop_list[15], sector_crop_list[48] ],
    
                    [ # threat zone 17
                      None, None, None, None, None, None, None, None,
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], 
                      sector_crop_list[4], sector_crop_list[4] ] ]


#----------------------------------------------------------------------------------
# read_header(infile):  takes an aps file and creates a dict of the data
#
# infile:               an aps file
#
# returns:              all of the fields in the header
#----------------------------------------------------------------------------------
def read_header(infile):
    # declare dictionary
    h = dict()
    
    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h
 
 #----------------------------------------------------------------------------------
# read_data(infile):  reads and rescales any of the four image types
#
# infile:             an .aps, .aps3d, .a3d, or ahi file
#
# returns:            the stack of images
#
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16      
#----------------------------------------------------------------------------------

def read_data(infile):
    
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    
    extension = os.path.splitext(infile)[1]
    
    with open(infile, 'rb') as fid:
          
        # skip the header
        fid.seek(512) 

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':
        
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): 
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor'] 
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':
              
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
                
            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy() 
            
        # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0,:,:,:].copy()
            imag = data[1,:,:,:].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag

#----------------------------------------------------------------------------------------
# getDF(infile): get the data frame for the patient, zone, and probabilities
#
# infile:                      labels csv file
#
# returns:                     a dataframe from the CS Document 
#
#----------------------------------------------------------------------------------------

def getDF(infile):
    # pull the labels for a given patient
    df = pd.read_csv(infile)

    # Separate the zone and patient id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_Zone',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    
    return df

#----------------------------------------------------------------------------------
# plot_image_set(infile):  takes an aps file and shows all 16 90 degree shots
#
# infile:                  an aps file
#----------------------------------------------------------------------------------
def plot_image_set(infile):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()
        
    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    
    i = 0
    for row in range(4):
        for col in range(4):
            resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
            axarr[row, col].imshow(np.flipud(img), cmap=COLORMAP)
            i += 1
    
    print('Done!')


#----------------------------------------------------------------------------------
# get_single_image(infile, nth_image):  returns the nth image from the image stack
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------

def get_single_image(infile, nth_image):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()
    
    return np.flipud(img[nth_image])


#----------------------------------------------------------------------------------
# convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------
def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)

#-------------------------------------------------------------------------------
# spread_spectrum(img):        applies a histogram equalization transformation
#
# img:                         a single scan
#
# returns:                     a transformed scan
#-------------------------------------------------------------------------------

def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)
    
    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img

#-----------------------------------------------------------------------------------------
# crop(img, crop_list):                uses vertices to mask the image
#
# img:                                 the image to be cropped
#
# crop_list:                           a crop_list entry with [x , y, width, height]
#
# returns:                             a cropped image
#-----------------------------------------------------------------------------------------
def crop(img, crop_list):

    x_coord = crop_list[0]
    y_coord = crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]
    #plt.imshow(cropped_img)
    #plt.colorbar()
    #plt.show(block=False)
    return cropped_img

#------------------------------------------------------------------------------------------
# normalize(image): Take segmented tsa image and normalize pixel values to be 
#                   between 0 and 1
#
# parameters:      image - a tsa scan
#
# returns:         a normalized image
#
#------------------------------------------------------------------------------------------

def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#-------------------------------------------------------------------------------------
# zero_center(image): Shift normalized image data and move the range so it is 0 c
#                     entered at the PIXEL_MEAN
#
# parameters:         image
#
# returns:            a zero centered image
#
#-----------------------------------------------------------------------------------------------------------
def zero_center(image):
     
    PIXEL_MEAN = 0.014327
    
    image = image - PIXEL_MEAN
    return image


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets for trainer 1
# this trainer is trained on 222x222 snapshots of the target 
# changed:    df to use 
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------
def preprocess_tsa_data(LABELS=TEST_LABELS):
    
    #Get the labels for each subject
    df = getDF(LABELS)
    SUBJECT_LIST = df['Subject'].unique()

    # intialize tracking and saving item
    batch_num = 1
    start_time = timer()

    #for each of the subjects
    for subject in SUBJECT_LIST:
        #open directory for aps files and get file for subject
        my_file = Path(I_FOLDER + '/' + subject + '.aps')
        if my_file.is_file():
            # read in the images
            print('--------------------------------------------------------------')
            print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, subject))
            print('--------------------------------------------------------------')
            images = read_data(I_FOLDER + '/' + subject + '.aps')
            # transpose so that the slice is the first dimension shape(16, 620, 512)
            images = images.transpose()
                       
            # get subjects labels
            df = getDF(LABELS)
            df = df.where(df['Subject'] == subject)
                       
            # for each threat zone, loop through each image, mask off the zone and then crop it
            for tz_num, threat_zone_x_crop_dims in enumerate(zip(zone_slice_list, zone_crop_list)):
                 
                
                threat_zone = threat_zone_x_crop_dims[0]
                crop_dims = threat_zone_x_crop_dims[1]
                zone_Num_String = str(tz_num + 1)
                       
                # get probability label if from training labels
                if(LABELS == TRAIN_LABELS): 
	                df = getDF(LABELS)
	                df = df.where(df['Subject']== subject)
	                df = df.where(df['Zone'] == zone_Num_String)
	                df.to_string(index=False) 
	                p = df[df.Zone == zone_Num_String].Probability.item()

                count = 0
                array_of_files= []
                for img_num, img in enumerate(images):
                    if(threat_zone[img_num] is not None):
                        # correct the orientation of the image
                        print('-> reorienting base image') 
                        base_img = np.flipud(img)
                        print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))
                        print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                        #plt.imshow(base_img)
                        #plt.colorbar()
                        #plt.show(block=False)
    
                        # convert to grayscale
                        print('-> converting to grayscale')
                        rescaled_img = convert_to_grayscale(base_img)
                        print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                      rescaled_img.mean()))

                        #spread the spectrum to improve contrast
                        print('-> spreading spectrum')
                        high_contrast_img = spread_spectrum(rescaled_img)
                        print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))
                    
                        # crop the image
                        print('-> cropping image')
                        print(crop_dims[img_num])
                        cropped_img = crop(high_contrast_img, crop_dims[img_num])
                        print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))


                        # normalize the image
                        print('-> normalizing image')
                        normalized_img = normalize(cropped_img)
                        print('-> shape {}|mean={}'.format(normalized_img.shape, normalized_img.mean()))
                        
                        # create descriptive file name
                        if(LABELS == TEST_LABELS):
                        	#the test file names do not have any description of the threat probability
                        	file_name = subject + '_Zone' + zone_Num_String + '_' + 'Img'+ str(img_num)+".jpg"
                        else: 
                        	#the training file names include info about the threat probability
                        	file_name = subject + '_Zone' + zone_Num_String + '_' + 'Img'+ str(img_num) +  '_' + str(p) +".jpg"
                        
                        im = Image.fromarray(normalized_img*255)
                        im = im.convert('RGB')

                        # determine correct path for image
                        if(LABELS == TEST_LABELS):
                        	file_path = TEST_PATH + file_name
                        elif(str(p)== '0.0'): 
                            file_path = TRAIN_PATH + "nothreats/" + file_name
                        else:
                            file_path = TRAIN_PATH + "threats/" + file_name

                        im.save(file_path, "JPEG")
                            
                        count = count + 1
                        array_of_files.append(file_path)
                            
                result = Image.new("RGB", (222, 222))
                for index, file in enumerate(array_of_files):
                    path = os.path.expanduser(file)
                    img = Image.open(path)
                    img.thumbnail((111, 111), Image.ANTIALIAS)
                    x = index // 2* 111
                    y = index % 2 * 111
                    w, h = img.size
                    print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
                    result.paste(img, (x, y, x + w, y + h))
                    os.remove(file)

                
                
                if(LABELS== TEST_LABELS):
                	file_name = subject + '_Zone' + zone_Num_String +".jpg"
                	file_path = TEST_PATH + file_name   
                elif(str(p)== '0.0'):
                	file_name = subject + '_Zone' + zone_Num_String + '_' + 'Img_Comb' +  '_' + str(p) +".jpg"
                	file_path = TRAIN_PATH + "nothreats/" + file_name
                else:
                	file_name = subject + '_Zone' + zone_Num_String + '_' + 'Img_Comb' +  '_' + str(p) +".jpg"
                	file_path = TRAIN_PATH + "threats/" + file_name
                result.save(os.path.expanduser(file_path))
# ---------------------------------------


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): grabs several files from the train/threats and train/nothreats 
# directories and moves them to the val/threats and val/notreats directories
#---------------------------------------------------------------------------------------
def split_train_data():
	# move no threats 
	path = TRAIN_PATH + 'nothreats/'
	os.chdir(path)
	#%cd /home/rg0407581/courses/deeplearning1/data/tsa/preprocessed_4/train/nothreats
	g = glob('*.jpg')
	shuf = np.random.permutation(g)
	for i in range(5000): os.rename(shuf[i], VAL_PATH +'nothreats/' + shuf[i])
	# MOVE threats
	path = TRAIN_PATH + 'threats/'
	os.chdir(path)
	g = glob('*.jpg')
	shuf = np.random.permutation(g)
	for i in range(550): os.rename(shuf[i], VAL_PATH +'threats/' + shuf[i])


if __name__ == '__main__':
	if(sys.argv[1] == "-test"):
		if(len(sys.argv) > 2): 
			preprocess_tsa_data(DIR_PATH + '/' + sys.argv[2])
		else: 
			preprocess_tsa_data(TEST_LABELS)
	elif(sys.argv[1] == "-train"): 
		preprocess_tsa_data(TRAIN_LABELS)
		split_train_data()
	else:
		print("Rerun this program with '-train' or '-test' to preprocess tsa data")
