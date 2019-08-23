# import libraries
import numpy as np
import matplotlib.pyplot as plt
import collections

def plotbar(dct, method, cluster, precision, pop=False):

    # Define feature names
    Col_names=[ "orig_sales_order_line_num", "unit_price_amt", "item_wt_qty", "order_qty", "price_match_amt", "promo_amt", "po_line_num", "item_wt_uom_cd", "fulfmt_type_desc"]
 
    Feature_names= ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'] 

    textstr=str()
    feature_iter=[]
          
    # define list with feature description
    for j in range(len(Feature_names)):
         feature_iter.append( (Feature_names[j] + '='+ Col_names[j] ))                       

    #
    textstr = '\n'.join(
                     (  feature_iter[0],
                        feature_iter[1], 
                        feature_iter[2],
                        feature_iter[3],
                        feature_iter[4],
                        feature_iter[5],
                        feature_iter[6],
                        feature_iter[7],
                        feature_iter[8]
                      )
                    )

    #eval_meth = [ 'weight', 'gain', 'cover', 'total_cover', 'total_gain' ] 
    eval_meth=['gain']
    #dct=bst.get_score(importance_type=str(method)) 
    if pop is True:
       for i in range(len(Col_names)):
           if Feature_names[i] in dct:
              dct[ Col_names[i] ] = dct.pop( Feature_names[i] )

    sorted_dct = sorted(dct.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dct)
    #print( list(sorted_dict.values() ) )
    #print( list(sorted_dict.values() ) / np.sum( list(sorted_dict.values() ) ) )  
    print( list(sorted_dict.keys() ) )

    #
    plt.bar(sorted_dict.keys(), (list(sorted_dict.values() ) / np.sum(list(sorted_dict.values() ))*100   ), color='b')
    plt.xlabel('Feature')
    plt.ylabel('Importance (%)')
    plt.title('Feature Importance using: ' + str(method) + ', for cluster: ' + str(cluster) + '\n with precision: ' + str(round(precision,3)*100) + str('%') ) 

    ax=plt.axes()
    ax.grid(which='both', axis='y', linestyle='--')
    #ax.yaxis.grid(True)

    plt.xticks(rotation=0)
    plt.show()




