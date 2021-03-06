import cv2
from skimage import color, data, restoration
import numpy as np
import sys
EPSILON=sys.float_info.epsilon

def f(q,a=1.62,c=0.6,func=None):
    if func is not None:
        return func(q)
    q_a=q**a
    return np.power((q_a/(1+q_a)),c)

def f_inverse(f,a=1.62,c=0.6,func=None):
    if func is not None:
        return func(f)

    f_cth_root=np.power(f,1/c)
    return np.power( f_cth_root/(1-f_cth_root) ,1/a)


def preprocess(image):
    # normalize pixels to [0,1] range
    image = image / 255.
    image[image >= 1] = 1 - EPSILON
    return image

def deblur(image,n=5,m=25):
    psf = np.ones((n, n)) / m
    B = image.reshape(image.shape[0],image.shape[1],3)[:,:,0]
    G = image.reshape(image.shape[0],image.shape[1],3)[:,:,1]
    R = image.reshape(image.shape[0],image.shape[1],3)[:,:,2]
    deconvolved_B = restoration.wiener(B, psf,1,clip=False)
    deconvolved_G = restoration.wiener(G, psf,1,clip=False)
    deconvolved_R = restoration.wiener(R, psf,1,clip=False)
    rgbArray = np.zeros((image.shape[0], image.shape[1], 3))
    rgbArray[..., 0] = deconvolved_B
    rgbArray[..., 1] = deconvolved_G
    rgbArray[..., 2] = deconvolved_R
    #print(rgbArray)
    #rgbArray[rgbArray < 0] = 0
    #rgbArray[rgbArray >= 1] = 1 - EPSILON
    return rgbArray

blur = cv2.imread('deblur/2.jpg')
blur = preprocess(blur)
cv2.imshow('original blur img',blur)
normal_deblur = deblur(blur)
cv2.imshow('normal deblur img',normal_deblur)
lightspace_deblur =f_inverse(blur)
cv2.imwrite('lightspace.jpg',lightspace_deblur*255)
#print(f(f_inverse(blur)))
cv2.imshow('lightspace deblur img',lightspace_deblur)
good = cv2.imread('good.jpg')
good = preprocess(good)
good = f(good)
print(good)
cv2.imshow('good',good)
cv2.waitKey(0)
cv2.destroyAllWindows()