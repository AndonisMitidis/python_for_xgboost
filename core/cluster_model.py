# import python libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class Cluster_Model():
       """A class for building clusters"""

       def __init__(self):
            self.model = KMeans

       def load_model(self, filepath):
            print('[Model] Loading model from file %s' % filepath)
            self.model = load_model(filepath)

       def run_clustering(self, ):

           top_stores_top_depts = pd.read_csv("/Users/vn50c26/Documents/NilPicks/top_stores_all_depts_nil_picks.csv")
           top_depts = pd.read_csv("/Users/vn50c26/Documents/NilPicks/top_depts_nil_picks.csv")
           dept_names = pd.read_csv("/Users/vn50c26/Documents/NilPicks/depts.csv")

           top_dept_names = pd.merge(top_depts, dept_names, on = "dept_nbr")

           top_dept_names['total_nil_pks'] = top_dept_names['nil_pick_cnts'].sum()
           top_dept_names['total_vol'] = top_dept_names['total_vol_by_dept'].sum()
           top_dept_names['nil_pk_dept_perc'] = np.round(top_dept_names['nil_pick_cnts']/top_dept_names['total_vol_by_dept']*100,3)
           top_dept_names['total_vol_dept_perc'] = np.round(top_dept_names['total_vol_by_dept']/top_dept_names['total_vol']*100,3)
           top_dept_names['nil_pick_perc'] = np.round(top_dept_names['nil_pick_ratio']*100,3)
           top_dept_names.drop(['total_nil_pks','total_vol','nil_pick_ratio'], axis = 1, inplace = True)

           dept_th = top_dept_names.loc[top_dept_names['total_vol_dept_perc']>3,'dept_nbr'].values.tolist()

           stores = pd.merge(top_stores_top_depts[top_stores_top_depts['dept_nbr'].isin(dept_th)], dept_names, on="dept_nbr", how="inner")
           stores['total_nil_pks'] = stores['nil_pick_cnts'].sum()
           stores['total_vol'] = stores['total_vol_by_dept'].sum()

           stores = pd.merge(stores, stores.groupby('store_id').apply(lambda x: np.round((100 * x["nil_pick_cnts"].sum()/x["total_vol_by_dept"].sum()),3))\
                    .reset_index(name='nil_pk_st_perc')
                    , on = "store_id")

           stores = pd.merge(stores, stores.groupby(['dept_nbr','dept_name']).apply(lambda x: np.round((100 * x["nil_pick_cnts"].sum()/x["total_vol_by_dept"].sum()),3))\
                    .reset_index(name='nil_pk_dept_perc')
                    , on = ['dept_nbr','dept_name'])

           stores['nil_pick_perc'] = np.round(stores['nil_pick_ratio']*100,3)
           stores.drop(['nil_pick_cnts','total_vol_by_dept','nil_pick_ratio','total_nil_pks','total_vol'], axis = 1, inplace = True)

           stores_prep = stores[['store_id','dept_nbr','nil_pick_perc']]
           stores_prep['dept_nbr'] = stores['dept_nbr'].apply(lambda x: 'd_' + str(x))
           stores_prep.set_index('store_id', inplace=True)

           #del stores_prep2
           stores_prep2 = pd.pivot_table(stores_prep, values = 'nil_pick_perc', index=['store_id'], columns = 'dept_nbr')

           mms = MinMaxScaler()
           mms.fit(stores_prep2)
           stores_prep2_sc = mms.transform(stores_prep2)

           Sum_of_Sq_Dist = []
           K = range(1,21)

           for k in K:
               km = KMeans(n_clusters=k)
               km = km.fit(stores_prep2)
               Sum_of_Sq_Dist.append(km.inertia_)

           Sum_of_Sq_Dist_Df = pd.DataFrame(Sum_of_Sq_Dist)
           Sum_of_Sq_Dist_Df["Cluster"] = Sum_of_Sq_Dist_Df.index + 1
           Sum_of_Sq_Dist_Df.columns = ['Value','Cluster']

           #(ggplot(Sum_of_Sq_Dist_Df)
           #    + geom_point(aes(Sum_of_Sq_Dist_Df["Cluster"], Sum_of_Sq_Dist_Df["Value"]))
           #    + geom_line(aes(Sum_of_Sq_Dist_Df["Cluster"], Sum_of_Sq_Dist_Df["Value"]))
           #    + xlab("Number of Clusters")
           #    + ylab("Sum of Squared Distances")
           #    + ggtitle("Elbow Method for Optimum Clusters"))
   
           # Initializing KMeans
           kmeans = KMeans(n_clusters=9, max_iter=100, n_init=1, verbose=0, random_state=3425)

           # Fitting with inputs
           kmeans = kmeans.fit(stores_prep2_sc)
           # Predicting the clusters
           labels = kmeans.predict(stores_prep2_sc)
           # Getting the cluster centers
           C = kmeans.cluster_centers_
           #print(labels)
           #print(C)

           stores_prep2["cluster"] = labels+1
           stores_prep2["cluster"].value_counts(sort=False).tolist()

           store_cluster = stores_prep2[["cluster"]].reset_index(level=['store_id'])
           clust_store_list = store_cluster.groupby('cluster')['store_id'].apply(lambda x: ", ".join(x.astype(str)))





