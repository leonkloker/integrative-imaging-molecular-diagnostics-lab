import cv2 as cv
import numpy as np
import pandas as pd
import os
import sys

if len(sys.argv) < 3: 
    print('Usage:')
    print('$python3 {} <directory of cell data files> <transformation parameters file>'.format(sys.argv[0]))
    sys.exit(0)

# directory of the cell data
cell_data_dir = sys.argv[1]
if cell_data_dir[-1] != '/':
     cell_data_dir += '/'

# file containing transformation parameters
parameter_file = sys.argv[2]

# new directory for transformed cell data
cell_data_new_dir = cell_data_dir[:-1] + "_coregistered"

#if not os.path.exists(cell_data_new_dir):
#        os.makedirs(cell_data_new_dir)

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
    angle =  -2 * np.pi * (90 + float(parameters['angle'])) / 360

    src = cv.imread('TMA_cores_M06_M07_panels/M06/Cores/'+file_name+'.png')
    
    # load original x, y coordinates from file
    x = np.array(cell_data.loc[:,'Cell X Position']).reshape((1,-1)).astype(np.uint16)
    y = np.array(cell_data.loc[:,'Cell Y Position']).reshape((1,-1)).astype(np.uint16)

    ###
    ### Coordinate Transformation
    ###

    # define size of the underlying canvas
    canvas_x = src.shape[1]
    canvas_y = src.shape[0]

    # set smallest values to 0
    x = x - np.min(x)
    y = y - np.min(y)

    # scale image up to the size of the canvas + additional scaling
    x = x * (canvas_x + w) / np.max(x)
    y = y * (canvas_y + h) / np.max(y)

    # crop points that are out of canvas
    x[x>src.shape[0]] = np.nan
    y[x>src.shape[0]] = np.nan
    x[y>src.shape[1]] = np.nan
    y[y>src.shape[1]] = np.nan

    # shift all points
    x = x + shift_w
    y = y - shift_h

    # crop points that are out of canvas
    x[x>src.shape[0]] = np.nan
    y[x>src.shape[0]] = np.nan
    x[y>src.shape[1]] = np.nan
    y[y>src.shape[1]] = np.nan
    x[x<0] = np.nan
    y[x<0] = np.nan
    x[y<0] = np.nan
    y[y<0] = np.nan

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

    ###
    ### End of Coordinate Transformation
    ###

    """res = np.ones((int(np.max(y)), int(np.max(x)), 3), np.uint8)

    for xi, yi in zip(x[0],y[0]):
        cv.circle(res, (int(xi),int(yi)), radius=5, color=(255, 0, 0), thickness=-1)

    res = cv.resize(res, (int(canvas_x + w), int(canvas_y + h)))

    M = np.float32([[1,0,shift_w],[0,1,-shift_h]])
    res = cv.warpAffine(res, M, (res.shape[1],res.shape[0]))

    M = cv.getRotationMatrix2D(((src.shape[1]-1)/2.0, (src.shape[0]-1)/2.0), 90 + angle, 1)
    res = cv.warpAffine(res, M, (res.shape[1],res.shape[0]))

    """

    for xi, yi in zip(x[np.isnan(x)==0], y[np.isnan(y)==0]):
        cv.circle(src, (int(xi),int(yi)), radius=5, color=(255, 0, 0), thickness=-1)
    """print(src.shape, res.shape)
    src = 0.5*src + 0.5*res[:src.shape[0],:src.shape[1]]"""
    
    if not os.path.isdir("coregistered_cores"):
        os.mkdir("coregistered_cores")

    cv.imwrite("coregistered_cores/"+file_name+'.png', src)

    #cell_data.loc[:, 'Cell X Position'] = x
    #cell_data.loc[:, 'Cell Y Position'] = y

    #cell_data.to_csv(cell_data_new_dir + "/" + data_file)

if not_found > 0:
     print("The transformation parameters for {} file(s) were not found in {} !".format(not_found, parameter_file))
