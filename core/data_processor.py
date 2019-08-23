# import python libraries
import os
import numpy as np
import pandas as pd
import pickle as pk
import xgboost as xgb
from sklearn.model_selection import train_test_split

# import modules from utils
from utils import equal_class 

class DataLoader():
    '''A class for loading and transforming data for the nil-pick models '''

    def __init__(self, dataframe):
        self.df=dataframe           

    def preprocess_numerical(self, cols):
        proc_numer_data = []
        for col in cols:       
            max_ = np.max(self.df[col]) 
            min_ = np.min(self.df[col]) 
            print(min_, max_)
            normalised_col = [ (float(p-min_) / float(max_-min_) ) for p in self.df[col] ] 
            proc_numer_data.append(normalised_col)
        proc_numer_data=np.array(proc_numer_data).T         
        print(proc_numer_data.shape)
        return np.array(proc_numer_data)

    def preprocess_categorical(self, cols): 
        proc_cat_data=[]      
        for col in cols:
            uniques=self.df[col].unique() 
            print(uniques) 
            np.savetxt('./info_from_preproc/uniques/'+str(col), uniques, newline='\n', fmt='%s' )
            CC=[]
            for i in range(len(uniques)):
                print('processing col: ', col, 'for unique value: ', uniques[i]) 
                for j in range(self.df.shape[0]):
                    if self.df[col][j] == uniques[i]:
                       CC.append(i/float(len(uniques)-1) )
            proc_cat_data.append(CC)
        proc_cat_data=np.array(proc_cat_data).T         
        print(proc_cat_data.shape)
        return np.array(proc_cat_data)

    def preprocess_target(self, cols): 
        proc_target_data=[]      
        a=self.df[cols[0]]
        print(a.unique())
        for i in range(self.df.shape[0]):
            if a[i]=='NILPICK' or a[i]=='SUBS':
                 proc_target_data.append(1)
            elif a[i]=='PICK' or a[i]=='ACTUALPICK' or a[i]=='OVRRIDE':
                 proc_target_data.append(0)
            else:
                 proc_target_data.append(-1)
        proc_target_data=np.array(proc_target_data).T         
        print(proc_target_data.shape)
        print(np.unique(proc_target_data))
        proc_target_data=proc_target_data.reshape((len(proc_target_data),1))
        return np.array(proc_target_data)

    def create_train_dct(self, train_array, clusters):
        dct = {}
        dct_A={}
        for i in range(len(clusters)):                                  # loop over list of clusters
            print('Creating cluster ', i)
            dct['uni_%s' % i] = []                                      # initiate a list for each set of clusters
            dct_A['uni_%s' % i] = []
            for j in range(self.df.shape[0]):                           # loop over all data rows of feature "store_id"
                if self.df["store_id"][j] in clusters[i]:               # if store_id value in data row is in cluster 'i'    
                   dct['uni_%s' % i].append(train_array[j])             # then, add store_id in the ith key of the dct
                   dct_A['uni_%s' % i].append(j)               
            print('the length of cluster: ', i, 'is: ', len(dct_A['uni_%s' % i]) )
        with open('./info_from_preproc/cluster_dictionary.pkl', 'wb') as f:
             pk.dump(dct, f, pk.HIGHEST_PROTOCOL)
        return dct
 
    def read_dct(self, dictionary, cluster_number, limit):             # read data for a particular cluster 
        print('This is cluster: ', cluster_number)                     # limit the number of rows in cluster data
        X=np.array(dictionary['uni_%s'%cluster_number])                # Define the data for cluster of number "cluster number"
        X=X[:limit,:]                                                  # Limit the number of rows
        new_data, new_target = equal_class.equal(X)                    # call utils function to create 'equal_class'
        new_data=np.array(new_data)
        new_target=np.array(new_target)
        new_target=new_target.reshape((len(new_target), 1))                
        print(new_data.shape, new_target.shape)

        X=np.hstack([new_data, new_target])
        np.random.shuffle(X)                                           # Shuffle the rows of that particular lane
        print(X.shape)

        Train=X[:, :-1]                                                # Define train and target values
        Train_T=X[:, -1:]
        print(Train.shape, Train_T.shape)                              # print dimensions for check
        X_train, X_test, y_train, y_test = train_test_split(Train, Train_T, test_size=0.2, random_state=42)   #split data to train and test sets

        return X_train, X_test, y_train, y_test




