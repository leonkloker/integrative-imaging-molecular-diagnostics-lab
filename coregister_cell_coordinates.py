import argparse
import cv2 as cv
import numpy as np
import pandas as pd
import os
import sys

# define input arguments
parser = argparse.ArgumentParser(description = "Argument reader")
parser.add_argument("-tf", "--transformation_parameter_file", help = "CSV file containing the transformation parameters", required = True, default = "")
parser.add_argument("-d", "--cell_data_directory", help = "directory containing the cell data", required = True, default = "")
parser.add_argument("-i", "--save_images", help = "if specified, transformed images are saved", required = False, default = "")
parser.add_argument("-s", "--save_files", help = "if specified, transformed cell data are saved", required = False, default = "")

argument = parser.parse_args()

# get relevant files
parameter_file = argument.transformation_parameter_file
cell_data_dir = argument.cell_data_directory
if cell_data_dir[-1] != '/':
     cell_data_dir += '/'

# new directory for transformed cell data
cell_data_new_dir = cell_data_dir.split('/')[-1]
if cell_data_new_dir == "":
    cell_data_new_dir = cell_data_dir.split('/')[-2]
cell_data_new_dir = cell_data_new_dir + "_coregistered"

if not os.path.exists(cell_data_new_dir) and argument.save_files != "":
        os.makedirs(cell_data_new_dir)

# read tarsnformation parameters
parameter_df = pd.read_csv(parameter_file)

print("Transforming cell (x,y) coordinates of {} files...".format(len(os.listdir(cell_data_dir))))

not_found = 0

# iterate over all cell data files
for data_file in os.listdir(cell_data_dir):
    file_name = data_file[:-4]
    parameters = parameter_df.loc[parameter_df['File Name'] == file_name]
    cell_data = pd.read_csv(cell_data_dir + data_file)

    # check if transformation parameters for the file exist
    if parameters.empty:
        not_found += 1
        continue

    # load parameters from file
    h = float(parameters['h'])
    w = float(parameters['w'])
    shift_h = float(parameters['shift_height'])
    shift_w = float(parameters['shift_width'])
    angle = -2 * np.pi * float(parameters['angle']) / 360

    src = cv.imread('TMA_cores_M06_M07_panels/M06/Cores/'+file_name+'.png')
    
    # load original x, y coordinates from file
    x = np.array(cell_data.loc[:,'Cell X Position']).reshape((1,-1))
    y = np.array(cell_data.loc[:,'Cell Y Position']).reshape((1,-1))

    ###
    ### Coordinate Transformation
    ###

    # set smallest values to 0
    x = x - np.min(x)
    y = y - np.min(y)

    # define center of cells as rotation center
    hx = np.max(x)/2
    hy = np.max(y)/2

    # perform 90 degree rotation around center
    x = x - hx
    y = y - hy

    M = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],[np.sin(-0.5*np.pi), np.cos(-0.5*np.pi)]])
    xy = M@np.concatenate((x,y), axis=0)

    x = xy[0,:].reshape(1,-1)
    y = xy[1,:].reshape(1,-1)

    x = x + hx
    y = y + hy

    # reset origin
    x = x - np.min(x)
    y = y - np.min(y)

    # define size of the underlying canvas
    canvas_x = src.shape[1]
    canvas_y = src.shape[0]

    # scale image up to the size of the canvas + additional scaling
    x = x * (canvas_x + w) / np.max(x)
    y = y * (canvas_y + h) / np.max(y)

    # shift all points
    x = x + shift_w
    y = y - shift_h

    # define canvas center as rotation center
    hx = canvas_x/2
    hy = canvas_y/2

    # perform rotation around center
    x = x - hx
    y = y - hy

    M = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    xy = M@np.concatenate((x,y), axis=0)

    x = xy[0,:]
    y = xy[1,:]

    x = x + hx
    y = y + hy

    # crop cells outside of image
    y[x<0] = np.nan
    x[x<0] = np.nan
    x[y<0] = np.nan
    y[y<0] = np.nan
    y[x>canvas_x] = np.nan
    x[x>canvas_x] = np.nan
    x[y>canvas_y] = np.nan
    y[y>canvas_y] = np.nan
    
    ###
    ### End of Coordinate Transformation
    ###

    # save images if argument is specified
    if argument.save_images != "":
        for xi, yi in zip(x[np.isnan(x)==0], y[np.isnan(y)==0]):
            cv.circle(src, (int(xi),int(yi)), radius=5, color=(255, 0, 0), thickness=-1)
    
        if not os.path.isdir("coregistered_cores"):
            os.mkdir("coregistered_cores")

        cv.imwrite("coregistered_cores/"+file_name+'.png', src)

    # save transformed files if argument is specified
    if argument.save_files != "":
        cell_data.loc[:, 'Cell X Position'] = x
        cell_data.loc[:, 'Cell Y Position'] = y
        cell_data.to_csv(cell_data_new_dir + "/" + data_file)

if not_found > 0:
     print("The transformation parameters for {} file(s) were not found in {} !".format(not_found, parameter_file))
