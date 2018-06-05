import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
EPSILON=sys.float_info.epsilon

def sharpen(im):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im = cv2.filter2D(im, -1, kernel)
    return im

def f(q,a=1.682,c=0.522,func=None):
    if func is not None:
        return func(q)
    q_a=np.power(q,a)
    return np.power((q_a/(1+q_a)),c)

def f_inverse(f,a=1.682,c=0.522,func=None):
    if func is not None:
        return func(f)

    f_cth_root=np.power(f,1/c)
    return np.power( f_cth_root/(1-f_cth_root) ,1/a)

def edge_detection(im):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    im = cv2.filter2D(im, -1, kernel)
    return im


def preprocess(image):
    # normalize pixels to [0,1] range
    image = image / 255.
    image[image >= 1] = 1 - EPSILON
    return image

hdr_before = cv2.imread("out.jpg")
hdr_before = preprocess(hdr_before)
hdr_after_sharpen = f(sharpen(f_inverse(hdr_before)))
cv2.imshow('sharpen',hdr_after_sharpen)
cv2.imwrite('sharpen.jpg',hdr_after_sharpen*255)

hdr_after_edge = f(edge_detection(f_inverse(hdr_before)))
cv2.imshow('edge',hdr_after_edge)
cv2.imwrite('edge detection.jpg',hdr_after_edge*255)
cv2.waitKey(0)
cv2.destroyAllWindows()