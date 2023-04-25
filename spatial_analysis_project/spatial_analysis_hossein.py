# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:38:53 2023

@author: shmirjah
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import scipy
import scipy.integrate
import scipy.special

import os # for getting the dataset path
import glob # related to the path
import zipfile # for the augmentation
import functools
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from keras.models import load_model #Loads a model saved via model.save()
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (32,32)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import tensorflow as tf
#import tensorflow.contrib as tfcontrib
from keras import layers
from keras import losses
from keras import models
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, KFold



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input

import scipy.integrate
import scipy.special

import pickle 

import random as rng
import seaborn as sns
import umap
import umap.plot
import pandas as pd


#Panel="M06"
Panel="Both"


version="v10"

cell_names=['Lymphocyte', 'Macrophage', 'Neutrophil', 'Others', 'Tumor']

cell_names_ratio=[]
for i, item in enumerate(cell_names):
    for j in cell_names[i+1:]:
        cell_names_ratio=cell_names_ratio+[item+"_over_"+j]


cell_names_neighbor=['Adjacent_'+i for i in cell_names]  

cell_min_neighbor=['Adjacent_Min_'+i for i in cell_names]  
  
cell_names_neighbor.insert(0, 'Name') # Adding an element to the list from left    

prognostic_direcotry="C:/Stanford_Projects/Prognostic Data/"


prediction_directory_M06="C:/Stanford_Projects/mxIF Deep Learning_Clustering_GMM_All/TMA/Results/M06/Predicted Texts/"
prediction_directory_M07="C:/Stanford_Projects/mxIF Deep Learning_Clustering_GMM_All/TMA/Results/M07/Predicted Texts/"
prognostic_data=pd.read_csv("C:/Stanford_Projects/Prognostic Data/Patient_Prognostic_Information_v2.csv")


#max_distance=100 # the window's radius within which we are scanning 
max_distance=50

features_core_level=pd.DataFrame(columns=['Core ID']+cell_names+cell_names_ratio, index=None)

features_cells_level_groupped=[]


# Loop over all patients:
    
for i,core_name in enumerate(prognostic_data["Core ID"]):
    
    total_patients=len(prognostic_data["Core ID"])
    
    print(f"for {core_name}: {i+1}/{total_patients}")
 

    pred_data_wt_features=pd.DataFrame(index=None)
    
# For Panel M06

    
    
    M06_pred_data_wt_features=pd.DataFrame(index=None)

    if core_name+".csv" in os.listdir(prediction_directory_M06): # check whether the core exists in files or not
            
        
        features_cell_level=pd.DataFrame()
        
        M06_pred_data=pd.read_csv(prediction_directory_M06+core_name+".csv")
        
        cells_type, cells_count=np.unique(M06_pred_data["Class Type"].to_numpy(), return_counts=True)
        #cells_type=list(cells_type)
        
        Set_cells_type=set(cells_type)
        Set_cell_names=set(cell_names)
        
        Set_difference=Set_cell_names.difference(Set_cells_type) # find the difference between these two sets, if there is any we need to append to the list and assign its count to 0
        
        
        pd_features_core=pd.DataFrame()
        M06_core_features=[0]*len(cell_names)
        #M06_core_features.append('M06')
        
        for k, cell in enumerate(cell_names):
            if cell in cells_type:                
                M06_core_features[k]=float(cells_count[np.where(cells_type==cell)][0])
                
        
        ## Warning: This caclulation needs correction when two panels are considered. The number of cells belonging to all cores of a patient need to be calculated, summed, and then obtain ratios
        
        M06_core_ratio_features=[0]*10
        
        m=0
        for l in range(len(M06_core_features)):
            for value in M06_core_features[l+1:]:
                if value <1:
                    value=1
                M06_core_ratio_features[m]=M06_core_features[l]/value
                m +=1
        
        
        M06_core_features.insert(0,core_name)
        M06_core_features=M06_core_features+M06_core_ratio_features
        
        features_core_level.loc[len(features_core_level.index)]=M06_core_features
        
        
        Cell_Locations=np.zeros((M06_pred_data.shape[0],2))
        Cell_Locations[:,0]=M06_pred_data["Centroid X px"]
        Cell_Locations[:,1]=M06_pred_data["Centroid Y px"]
        
        
        features_all_cell_neighbor=[]
        
        for j in range(len(M06_pred_data["Centroid X px"])):
            
            features_cell_neighbor=[0]*len(cell_names)
            features_min_neighbor=[5000]*len(cell_names) 
            #features_cell_neighbor.append(M06_pred_data["Name"][j])
            
            cell=np.array([Cell_Locations[j,0],Cell_Locations[j,1]])
            dist=np.linalg.norm(cell - Cell_Locations[:,None], axis=-1) # calculating the Eculidean distance from one cell to others
            location_indexes=np.where((dist<max_distance) & (dist!=0))[0] # looking those within the window of close proximity, excluding the cell itself
            cell_type_neighbor=M06_pred_data["Class Type"][location_indexes]
            
            pd_cell_min_neighbor=pd.DataFrame()
            pd_cell_min_neighbor['Class Type']=M06_pred_data["Class Type"]
            #pd_cell_min_neighbor['Dist']=dist[location_indexes]
            pd_cell_min_neighbor['Dist']=dist
            pd_cell_min_neighbor=pd_cell_min_neighbor.loc[(pd_cell_min_neighbor!=0).all(axis=1)] # removing the cell with 0 distance (self distance)

            pd_cell_min_neighbor=pd_cell_min_neighbor.groupby('Class Type').min().reset_index()
            
            cell_type_min_neighbor=pd_cell_min_neighbor['Class Type'].to_numpy()
            cell_dist_min_neighbor=pd_cell_min_neighbor['Dist'].to_numpy()
            
            
            
            cells_type_neighbor, cells_count_neighbor=np.unique(cell_type_neighbor.to_numpy(), return_counts=True)
            
            
            Set_cells_type_neighbor=set(cells_type_neighbor)
            
            Set_difference=Set_cell_names.difference(Set_cells_type_neighbor) # find the difference between these two sets, if there is any we need to append to the list and assign its count to 0



            for m, cell in enumerate(cell_names):
                
                if cell in cells_type_neighbor:
                    features_cell_neighbor[m]=cells_count_neighbor[np.where(cells_type_neighbor==cell)][0]
                
                if cell in cell_type_min_neighbor:    
                    features_min_neighbor[m]=cell_dist_min_neighbor[np.where(cell_type_min_neighbor==cell)][0]

            
            features_cell_neighbor.insert(0,M06_pred_data["Name"][j])
            
            #features_all_cell_neighbor.append(features_cell_neighbor)
            features_cell_neighbor_combined=features_cell_neighbor+features_min_neighbor
            features_all_cell_neighbor.append(features_cell_neighbor_combined)
            

        df_features_all_cell_neighbor=pd.DataFrame(features_all_cell_neighbor, columns=cell_names_neighbor+cell_min_neighbor, index=None)        
        
        M06_pred_data_wt_features = pd.merge(M06_pred_data, df_features_all_cell_neighbor, how ='inner', on =['Name'])




# For Panel M07
    
    
    M07_pred_data_wt_features=pd.DataFrame(index=None)

    
    if core_name+".csv" in os.listdir(prediction_directory_M07): # check whether the core exists in files or not
            
        
        features_cell_level=pd.DataFrame()
        
        M07_pred_data=pd.read_csv(prediction_directory_M07+core_name+".csv")
        
        cells_type, cells_count=np.unique(M07_pred_data["Class Type"].to_numpy(), return_counts=True)
        #cells_type=list(cells_type)
        
        Set_cells_type=set(cells_type)
        Set_cell_names=set(cell_names)
        
        Set_difference=Set_cell_names.difference(Set_cells_type) # find the difference between these two sets, if there is any we need to append to the list and assign its count to 0
        
        
        pd_features_core=pd.DataFrame()
        M07_core_features=[0]*len(cell_names)
        #M07_core_features.append('M07')
        
        for k, cell in enumerate(cell_names):
            if cell in cells_type:                
                M07_core_features[k]=float(cells_count[np.where(cells_type==cell)][0])
                
        
        ## Warning: This caclulation needs correction when two panels are considered. The number of cells belonging to all cores of a patient need to be calculated, summed, and then obtain ratios
        
        M07_core_ratio_features=[0]*10
        
        m=0
        for l in range(len(M07_core_features)):
            for value in M07_core_features[l+1:]:
                if value <1:
                    value=1
                M07_core_ratio_features[m]=M07_core_features[l]/value
                m +=1
        
        
        M07_core_features.insert(0,core_name)
        M07_core_features=M07_core_features+M07_core_ratio_features
        
        features_core_level.loc[len(features_core_level.index)]=M07_core_features
        
        
        Cell_Locations=np.zeros((M07_pred_data.shape[0],2))
        Cell_Locations[:,0]=M07_pred_data["Centroid X px"]
        Cell_Locations[:,1]=M07_pred_data["Centroid Y px"]
        
        
        features_all_cell_neighbor=[]
        
        for j in range(len(M07_pred_data["Centroid X px"])):
            
            features_cell_neighbor=[0]*len(cell_names)
            features_min_neighbor=[5000]*len(cell_names) 
            #features_cell_neighbor.append(M07_pred_data["Name"][j])
            
            cell=np.array([Cell_Locations[j,0],Cell_Locations[j,1]])
            dist=np.linalg.norm(cell - Cell_Locations[:,None], axis=-1) # calculating the Eculidean distance from one cell to others
            location_indexes=np.where((dist<max_distance) & (dist!=0))[0] # looking those within the window of close proximity, excluding the cell itself
            cell_type_neighbor=M07_pred_data["Class Type"][location_indexes]
            
            pd_cell_min_neighbor=pd.DataFrame()
            pd_cell_min_neighbor['Class Type']=M07_pred_data["Class Type"]
            #pd_cell_min_neighbor['Dist']=dist[location_indexes]
            pd_cell_min_neighbor['Dist']=dist
            pd_cell_min_neighbor=pd_cell_min_neighbor.loc[(pd_cell_min_neighbor!=0).all(axis=1)] # removing the cell with 0 distance (self distance)

            pd_cell_min_neighbor=pd_cell_min_neighbor.groupby('Class Type').min().reset_index()
            
            cell_type_min_neighbor=pd_cell_min_neighbor['Class Type'].to_numpy()
            cell_dist_min_neighbor=pd_cell_min_neighbor['Dist'].to_numpy()
            
            
            
            cells_type_neighbor, cells_count_neighbor=np.unique(cell_type_neighbor.to_numpy(), return_counts=True)
            
            
            Set_cells_type_neighbor=set(cells_type_neighbor)
            
            Set_difference=Set_cell_names.difference(Set_cells_type_neighbor) # find the difference between these two sets, if there is any we need to append to the list and assign its count to 0



            for m, cell in enumerate(cell_names):
                
                if cell in cells_type_neighbor:
                    features_cell_neighbor[m]=cells_count_neighbor[np.where(cells_type_neighbor==cell)][0]
                
                if cell in cell_type_min_neighbor:    
                    features_min_neighbor[m]=cell_dist_min_neighbor[np.where(cell_type_min_neighbor==cell)][0]

            
            features_cell_neighbor.insert(0,M07_pred_data["Name"][j])
            
            #features_all_cell_neighbor.append(features_cell_neighbor)
            features_cell_neighbor_combined=features_cell_neighbor+features_min_neighbor
            features_all_cell_neighbor.append(features_cell_neighbor_combined)
            

        df_features_all_cell_neighbor=pd.DataFrame(features_all_cell_neighbor, columns=cell_names_neighbor+cell_min_neighbor, index=None)        
        
        M07_pred_data_wt_features = pd.merge(M07_pred_data, df_features_all_cell_neighbor, how ='inner', on =['Name'])


    









    
    ## Appending All cells belonging to the core(s) of a patient. This step is done after calculating spatial analysis
    
    #if core_name+".csv" in os.listdir(prediction_directory_M06):
    if core_name+".csv" in os.listdir(prediction_directory_M06) or core_name+".csv" in os.listdir(prediction_directory_M07):
    
        pred_data_wt_features=pred_data_wt_features.append(M06_pred_data_wt_features)
        pred_data_wt_features=pred_data_wt_features.append(M07_pred_data_wt_features)
        
        
        pred_data_wt_features=pred_data_wt_features.loc[:,"Class Type":]
        pred_data_wt_features_groupped=pred_data_wt_features.groupby('Class Type').mean().reset_index() # mean over those with similar Core ID
    
    
        
    
        Set_class_types=set(pred_data_wt_features_groupped["Class Type"])
        set_difference=list(Set_cell_names.difference(Set_class_types))
        
        
        for item in set_difference:
            empty_features=[]
            empty_features.append(item)
            empty_features=empty_features+[0]*len(cell_names)+[500]*len(features_min_neighbor)
            pred_data_wt_features_groupped.loc[len(pred_data_wt_features_groupped.index)]=empty_features
        
        pred_data_wt_features_groupped=pred_data_wt_features_groupped.sort_values(by="Class Type", axis=0)
        
        columns_fuetures=[i+"_"+pred_data_wt_features_groupped.columns[1:] for i in pred_data_wt_features_groupped["Class Type"]]
        
        import itertools
        columns_fuetures=list(itertools.chain(*columns_fuetures)) # flatten all elements
        columns_fuetures.insert(0, 'Core ID')
        
        reshaped_cells_features=pred_data_wt_features_groupped.loc[0:,"Adjacent_Lymphocyte":].to_numpy().reshape((1,50)).tolist()[0]
        reshaped_cells_features.insert(0, core_name)
        
        features_cells_level_groupped.append(reshaped_cells_features)
        



features_cells_level_groupped=pd.DataFrame(features_cells_level_groupped, columns=columns_fuetures, index=None)
features_core_level_groupped=features_core_level.groupby('Core ID').sum().reset_index() # mean over those with similar Core ID
#features_core_level_groupped=features_core_level

All_core_cell_features = pd.merge(features_core_level_groupped, features_cells_level_groupped, how ='inner', on =['Core ID'])

prognostic_data_wt_features=pd.merge(prognostic_data, All_core_cell_features, how='inner', on=['Core ID'])

prognostic_data_wt_features_density=prognostic_data_wt_features.copy()


core_cell_sums=prognostic_data_wt_features.loc[:,"Lymphocyte":"Tumor"].sum(axis=1).to_numpy() # summing over all cells for each core
df_sum=pd.DataFrame(core_cell_sums, columns=['Sum'], index=None) # converting to dataframe for the purpose of next step, division
#prognostic_data_wt_features_density.loc[:,"Lymphocyte":"Tumor"]=prognostic_data_wt_features.loc[:,"Lymphocyte":"Tumor"].div(df_sum.Sum, axis=0) # Cell proportion

prognostic_data_wt_features_density.loc[:,"Lymphocyte":"Tumor"]=prognostic_data_wt_features.loc[:,"Lymphocyte":"Tumor"].div(2.07) # Cell area density. 2.07mm^2 is the average area of a TMA


#prognostic_data_wt_features_density.to_csv(prognostic_direcotry+f"prognostic_data_wt_features_both_panels_{version}_{max_distance}pixels.csv", index=None)        
prognostic_data_wt_features_density.to_csv(prognostic_direcotry+f"prognostic_data_wt_features_{Panel}_{version}_{max_distance}pixels.csv", index=None)        
 

        
"""        
    if item in os.listdir(prediction_directory_M07):        
        M07_pred_data=pd.red_csv(prediction_directory_M07+item+".csv")
"""    
    
    


#MSC2390035736
#LIN2290282489 (Immigrant Petition for Alien Worker)

