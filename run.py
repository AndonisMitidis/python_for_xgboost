__author__ = 'Andonis Mitidis'
__version__ = '1.0.0'
__corp__ = 'Walmart'

import os
import os.path
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from core.data_processor import DataLoader
from core.xgb_model import XGB_Model
#from core.cluster_model import Cluster_Model

def main(do_preprocessing, do_clustering, do_dictionary, do_xgboost):

    #########################################################
    ### Load configs, Read csv, Create dataframe ############
    #########################################################
    configs = json.load(open('config.json', 'r'))                                                           # load json configuration file, that points to data csv file
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])          # 
    print('Reading csv file....')
    dataframe = pd.read_csv( os.path.join('data', configs['data']['filename']) )                            # load csv file
    print('Done!')
    data=DataLoader(dataframe)                                                                              # define the dataframe as 'data'
    
    #########################################################
    ### Conditions on preprocessing #########################
    #########################################################
    if do_preprocessing is True:                                                                              
        numerical_data   = data.preprocess_numerical(configs['data']['columns']['numerical'] )              # preprocess numerical data
        categorical_data = data.preprocess_categorical(configs['data']['columns']['categorical'])           # preprocess categorical data
        target           = data.preprocess_target(configs['data']['columns']['target'] )               # preprocess target data
        #time_stamps_data = data.preprocess_time_stamps_data(configs['data']['columns']['time_stamps'] )
        #multiplicity     = data.preprocess_multiplicity(configs['data']['columns']['multiplicity'] )    
        #store_id         = data.preprocess_store_id(configs['data']['columns']['store_id'] )
        #department_id    = data.preprocess_department_id(configs['data']['columns']['department_id'] )
        #category_nmbr    = data.preprocess_category_nmbr(configs['data']['columns']['category_nmbr'] )
        print(numerical_data.shape, categorical_data.shape, target.shape)
        train_array = np.hstack([numerical_data, categorical_data, target])                                 # stack the numer, categ, target pre-processed values 
        print(train_array.shape)
        np.savetxt('./info_from_preproc/train_array.txt', train_array, newline='\n', fmt='%s')              # create and save training array 
    else:
        print('preprocessing was not enabled in this run')

    #########################################################
    ### Conditions on clustering ############################
    #########################################################   
    clusters=[
              [535, 1201, 1597, 1833, 2307, 3223, 3265, 3352, 3777, 4240, 4274, 4318, 5186, 5707],
              [1178],
              [201, 212, 376, 537, 866, 1346, 1988, 2136, 2938, 3267, 3274, 3366, 3403, 4615, 5343, 5764],
              [69, 389, 532, 1342, 1626, 1875, 1885, 2111, 2266, 2511, 2966, 4299, 5728],
              [220, 398, 517, 994, 1156, 1341, 3261, 3295, 3431, 4438, 5141, 5841],
              [144, 259, 284, 1514, 1739, 2549, 2803, 3751, 5080, 5260, 5368, 5428],
              [16, 85, 125, 172, 277, 892, 963, 974, 990, 1125, 1440, 1902, 2300, 2587, 2685, 2862, 2980, 3017, 3209, 3283, 3773, 3835, 4333, 5234, 5241, 5313, 5703],
              [541, 953],
              [3802, 5247, 5884]
             ]
    if do_clustering is True:
        print('Keeping the clusters...') 
        #print('Running the clustering algorithm...')
        #cluster_model = Cluster_Model()                                                                    # Load the "Clustering_Model" class
        #clusters = cluster_model.run_clustering(       )                                                   # run_clustering algorithm and output clusters
        print('Done!')    
    else:
        new_clusters=[]
        for i in range(9): 
           new_clusters=new_clusters + clusters[i]
        clusters=new_clusters    

    #########################################################
    ### Conditions on dictionary ############################
    #########################################################    
    if do_dictionary is True:  
        print('Creating a dictionary based on the clusters formed by the clustering algorithm...')
        if do_preprocessing is False:
           print('Assuming you ran preprocessing before, you are now loading the existing train_array.txt file...')  
           train_array=np.loadtxt('./info_from_preproc/train_array.txt') 
           print('Done!')   
        dictionary = data.create_train_dct(train_array, clusters)                                           
    else:  
        print('Assuming a dictionary already exists, you are now loading the existing cluster_dictionary.pkl dictionary...')
        dictionary = joblib.load(open('./info_from_preproc/cluster_dictionary.pkl', 'rb'))                   
        print('Done!')

    #########################################################
    ### Conditions on xgboost ###############################
    #########################################################
    if do_xgboost is True:
        xgb_model = XGB_Model()                                                                       # Load the 'XGB_Model' class 
        for i in range(len(dictionary)):                                                              # loop over all clusters, nmbr of clusters = len(dct) 
            [X_train, X_test, y_train, y_test] = data.read_dct(dictionary, i, 200000)                 # run xgboost: train model, save model, predictions, plots 
            print('Training the xgboost classification algorihtm...')                                 # limit num of rows = 200000
            xgb_model.xgb_train(X_train, y_train, configs, 'gain', i)                                 # this saves the models as pickle files in saved models folder 
            print('Done!', ' Running the prediction based on a saved model...')
            predictions = xgb_model.xgb_pred(X_test, y_test, 'gain', i)
            print('Done!')
    else:
        print('xgboost was not enabled in this run.'

#############################################################
#### Choose Use Cases and edit main() accordingly ###########
#############################################################
if __name__ == '__main__':
    main(do_preprocessing=True, do_clustering=True, do_dictionary=True, do_xgboost=True)    
 
    # Use_Case 1. To preprocess a csv file:                           do_preprocessing=True,  do_clustering=False, do_dictionary=False, do_xgboost=False
    # Use_Case 2. To create clusters on an existing'train_array.txt'  do_preprocessing=False, do_clustering=True,  do_dictionary=False, do_xgboost=False 
    # Use_Case 3: To create a dictionary on a train_array.txt file:   do_preprocessing=False, do_clustering=False, do_dictionary=True,  do_xgboost=True
    # Use_Case 4. To run xgboost on an existing dictionary:           do_preprocessing=False, do_clustering=False, do_dictionary=False, do_xgboost=True
    # Use_Case 5: To create a dictionary on given clusters:           do_preprocessing=False, do_clustering=True,  do_dictionary=True,  do_xgboost=False
    # Use_Case 6. To pre-process, cluster, dictionary and xgboost:    do_preprocessing=True,  do_clustering=True,  do_dictionary=True,  do_xgboost=True
    # Note: If do_clustering=False then all clusters are merged to 1 cluster 

#############################################################
#############################################################


