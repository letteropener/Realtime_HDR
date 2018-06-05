import cv2
import numpy as np
#from PIL import Image
import sys
from timeit import default_timer as timer
from numba import vectorize,float64

@vectorize(["float64(float64)"],target='parallel')
def gamma(img):
    return np.power(img,(1/2.2))
if __name__=='__main__':
img = cv2.imread("2.jpg")
    start = timer()
    HDR_FRAME = (gamma(img/255) *255.).astype(np.uint8)
    end_time = timer() - start
    print("ldr_tonemapping took %f seconds/frame" % end_time)
    cv2.imwrite("HDR.jpg",HDR_FRAME)