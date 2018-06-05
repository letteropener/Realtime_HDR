import numpy as np
from scipy.stats import iqr

SIGMA_THRESHOLD=0.00001

def unroll_row(row):
    ret=[]
    for i in range(len(row)):
        count=row[i]
        if count>0:
            ret+=[i]*int(count)
    return np.array(ret)

def comparasum_row_iqr(comparasum,verbose=False):
    comparasum=np.asarray(comparasum)
    
    _iqr=[]
    for row in comparasum:
        #unroll row
        data=unroll_row(row)
        #print("data of row {} is\n {} with iqr".format(row,data))
        dist=iqr(data)
        #print(dist)
        _iqr.append(dist)
    if verbose is True:
        print("Current iqr",_iqr)
    


    return np.array(_iqr)
    