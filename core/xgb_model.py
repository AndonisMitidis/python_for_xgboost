# import python libraries
import os
import math
import numpy as np
import pickle as pk
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

import datetime as dt
from numpy import newaxis
from core.utils import Timer

# import modules from utils
from utils import plot_importance

class XGB_Model():
       """A class for building and inferencing an xgboost model"""

       def __init__(self):
            self.model = xgb

       def load_model(self, filepath):
            print('[Model] Loading model from file %s' % filepath)
            self.model = load_model(filepath)

       def xgb_train(self, X_train, y_train, configs, eval_meth, cluster_number):
            timer = Timer()
            timer.start()
            dtrain = self.model.DMatrix(X_train, label=y_train)            # create the Xgboost specific DMatrix data format from the numpy array            
            param = {                                                      # set xgboost params
                      'max_depth': configs['model']['max_depth'],          # the maximum depth of each tree
                            'eta': configs['model']['eta'],                # the training step for each iteration
                         'silent': configs['model']['silent'],             # logging mode - quiet
                      'objective': configs['model']['objective'],          # error evaluation for multiclass training
                      'num_class': configs['model']['num_class'],          # the number of classes that exist in this dataset
                     }                                                                             
            num_round = configs['model']['num_round']                      # the number of training iterations
            bst = self.model.train(param, dtrain, num_round)               # training and testing - numpy matrices  
            timer.stop()

            if not os.path.exists('./saved_models/cluster_'+str(cluster_number) ): 
               os.makedirs('./saved_models/cluster_'+str(cluster_number))
            bst.dump_model('./saved_models/cluster_'+str(cluster_number)+'/dump.raw.txt')                       # dump the models
            joblib.dump(bst,'./saved_models/cluster_'+str(cluster_number)+'/bst_model.pkl', compress=True)      # save the models for later

       def xgb_pred(self, X_test, y_test, eval_meth, cluster_number):
            dtest = self.model.DMatrix(X_test, label=y_test) # create the Xgboost specific DMatrix data format from the numpy array
            model=joblib.load(open('./saved_models/cluster_'+str(cluster_number)+'/bst_model.pkl', 'rb'))
            preds = model.predict(dtest)
            best_preds = np.asarray([np.argmax(line) for line in preds])   # extracting most confident predictions
            print("Numpy array precision:", precision_score(y_test, best_preds, average='macro') )
                 
            print('plotting...', eval_meth)                                # get importance scores based on following methods:  
            plt.subplot( 1, 1, 1)                                          # weight, gain, cover, total_gain, total_cover
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            dct=model.get_score(importance_type=str(eval_meth))

            #plt.text(5.5, 38, textstr, fontsize=10, verticalalignment='top', bbox=props)
            plot_importance.plotbar(dct, str(eval_meth), cluster_number, precision_score(y_test, best_preds, average='macro') )  # call utils function 'plot importance'
            print('plotting', eval_meth ,'done!')

            plt.show() 

      
