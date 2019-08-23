import numpy as np

def equal(data):

        C=data.shape[1]
	
        # Define target of all q
        target=data[:,C-1]
        #print target 

        # Create a training array with equal number of classes
        lp=[]
        lz=[]
        for i in range(len(target)):
            if target[i]==1:
               lp.append(i) 
            elif target[i]==0:
               lz.append(i)  
	
        print('Lengths of lp and lz:')
        print(len(lp) )
        print(len(lz),'\n')
	
        np.random.shuffle(lp)
        np.random.shuffle(lz)

        # Check what class is larger and cut the larger one to match the smaller one
        if len(lp)>len(lz): 
           lp=lp[:len(lz)]           
        elif len(lz)>len(lp): 
           lz=lz[:len(lp)]
	
        # Create final data array
        fin_data_arr_p=np.zeros((len(lp),C-1))
        fin_data_arr_z=np.zeros((len(lz),C-1))
        fin_data_arr=np.zeros((2*len(lp),C-1))
	
        #print fin_data_arr.shape
	
        for i in range(len(lp)):
            fin_data_arr_p[i]= data[lp[i],:-1]
        for i in range(len(lz)):
            fin_data_arr_z[i]= data[lz[i],:-1]    
        fin_data_arr=np.vstack([fin_data_arr_p, fin_data_arr_z])
	
        print('This is the shape of the final data array:')
        print(fin_data_arr.shape, '\n')
        #np.savetxt('./processed_col/fin_data_arr.txt', fin_data_arr, newline='\n', fmt='%1.3f' )
		
        # Create final target array
        fin_target_arr_p=np.zeros((len(lp),1))
        fin_target_arr_z=np.zeros((len(lz),1))
        fin_target_arr=np.zeros((2*len(lp),1))
	
        for i in range(len(lp)):
            fin_target_arr_p[i]= target[lp[i]]
        for i in range(len(lz)):
            fin_target_arr_z[i]= target[lz[i]]    
        fin_target_arr=np.vstack([fin_target_arr_p, fin_target_arr_z])
	
        print('This is the shape of the final target array:')
        print(fin_target_arr.shape, '\n')
        #np.savetxt('./processed_col/fin_target_arr.txt', fin_target_arr, newline='\n', fmt='%1.3f' )
	
        # Redefine data and target
        new_data=fin_data_arr
        target=fin_target_arr
        new_target=target.reshape((len(target),))				

        #print(new_data)
        #print(new_target) 
        new_data=list(new_data)
        new_target=list(new_target)
        return new_data, new_target   



