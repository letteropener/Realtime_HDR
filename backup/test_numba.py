import numpy as np
from numba import vectorize,guvectorize
import time

def test1(array1,array2):
    init_time = time.time()
    a = array1[100,100,1] + array2[100,200,0]
    print("test 1 time = ",time.time()-init_time)
    return a

@guvectorize(["uint8(uint8,uint8)"],'(n),(n)->(n)',target='parallel')
def test2(array1,array2):
    return array1 + array2[15000]

if __name__ == "__main__":
    array1 = np.zeros((300,400,2),np.uint8)
    array2 = np.ones((300,400,2),np.uint8)
    flat_a1 = array1.flatten()
    flat_a2 = array2.flatten()
    a1_shape = np.array([100,100,1],np.uint8)
    a2_shape = np.array([100,200,0],np.uint8)
    test1(array1,array2)
    init_time = time.time()
    #test2(flat_a1.tolist(),flat_a2.tolist(),a1_shape.tolist(),a2_shape.tolist())
    test(flat_a1,flat_a2)
    print("test 2 time = ",time.time()-init_time)