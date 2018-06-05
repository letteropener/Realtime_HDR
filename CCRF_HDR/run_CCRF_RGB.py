import cv2
import numpy as np
from PIL import Image
import sys
from timeit import default_timer as timer
from numba import vectorize

import os

os.environ['NUMBAPRO_NVVM']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\bin\nvvm64_31_0.dll'

os.environ['NUMBAPRO_LIBDEVICE']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\nvvm\libdevice'


CCRF_LUT_R = np.loadtxt('../LUTs/CCRF_R.txt')
CCRF_LUT_G = np.loadtxt('../LUTs/CCRF_G.txt')
CCRF_LUT_B = np.loadtxt('../LUTs/CCRF_B.txt')


def run_CCRF(f2q,fq):
    CCRF = np.zeros(fq.shape)
    for i in range(f2q.shape[0]):
        for j in range(f2q.shape[1]):
            X_B = fq[i][j][0]
            Y_B = f2q[i][j][0]
            CCRF[i][j][0] = CCRF_LUT_B[X_B][Y_B]
            X_G = fq[i][j][1]
            Y_G = f2q[i][j][1]
            CCRF[i][j][1] = CCRF_LUT_G[X_G][Y_G]
            X_R = fq[i][j][2]
            Y_R = f2q[i][j][2]
            CCRF[i][j][2] = CCRF_LUT_R[X_R][Y_R]
    return CCRF

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def GPU_CCRF_B(f2q_B,fq_B):
     return CCRF_LUT_B[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def GPU_CCRF_G(f2q_G,fq_G):
     return CCRF_LUT_G[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def GPU_CCRF_R(f2q_R,fq_R):
     return CCRF_LUT_R[fq_R][f2q_R]





if __name__ == "__main__":

    f2q = cv2.imread("16000.jpeg")
    fq = cv2.imread("8000.jpeg")
    parallel_start = timer()
    f2q_B = f2q[:,:,0]
    f2q_G = f2q[:,:,1]
    f2q_R = f2q[:,:,2]
    fq_B = fq[:,:,0]
    fq_G = fq[:,:,1]
    fq_R = fq[:,:,2]
    ccrf_B = GPU_CCRF_B(f2q_B,fq_B)
    ccrf_G = GPU_CCRF_G(f2q_G,fq_G)
    ccrf_R = GPU_CCRF_R(f2q_R,fq_R)
    ccrf = np.dstack((ccrf_B, ccrf_G, ccrf_R))
    parallel_time = timer() - parallel_start
    print("parallel CCRF Look Up took %f seconds/frame" % parallel_time)
    cv2.imwrite('ccrf.jpeg',ccrf)
    regular_start = timer()
    ccrf = run_CCRF(f2q,fq)
    regular_time = timer() - regular_start
    print("regular CCRF Look Up took %f seconds/frame" % regular_time)