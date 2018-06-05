import cv2
import numpy as np
import sys
EPSILON=sys.float_info.epsilon


def preprocess(image):
    # normalize pixels to [0,1] range
    image = image / 255.
    image[image >= 1] = 1 - EPSILON
    return image

def f(q,a=1.62,c=0.6,func=None):
    if func is not None:
        return func(q)
    q_a=np.power(q,a)
    return np.power((q_a/(1+q_a)),c)

def f_inverse(f,a=1.62,c=0.6,func=None):
    if func is not None:
        return func(f)

    f_cth_root=np.power(f,1/c)
    return np.power( f_cth_root/(1-f_cth_root) ,1/a)

v1 = cv2.imread('v12/v1.jpg')
v1 = preprocess(v1)
v2 = cv2.imread('v12/v2.jpg')
v2 = preprocess(v2)
v3 = cv2.imread('v12/v3.jpg')
v3 = preprocess(v3)

v_1_2 = (v1 + v2)
v_1_2[v_1_2 < 0] = 0
v_1_2[v_1_2 >= 1] = 1 - EPSILON

v_1_2_from_light = f(f_inverse(v1) + f_inverse(v2))
v_1_2_from_light[v_1_2_from_light < 0] = 0
v_1_2_from_light[v_1_2_from_light >= 1] = 1 - EPSILON
cv2.imshow('v_1_2',v_1_2)
v_1_2_save_img=(v_1_2*255.0).astype(np.int32)
cv2.imwrite("v12/v_1_2.jpg",v_1_2_save_img)
cv2.imshow('v_1_2_from_lightspace',v_1_2_from_light)
v_1_2_from_light_save_img=(v_1_2_from_light*255.0).astype(np.int32)
cv2.imwrite("v12/v_1_2_from_lightspace.jpg",v_1_2_from_light_save_img)
cv2.imshow('v1',v1)
cv2.imshow('v2',v2)
cv2.imshow('v3',v3)
num_pix=v_1_2.shape[0]*v_1_2.shape[1]

v_1_2_flat = v_1_2.reshape(num_pix*3)
v_1_2_from_light_flat = v_1_2_from_light.reshape(num_pix*3)
v3_flat = v3.reshape(num_pix*3)
print('mean square error (V12 with V3): ',sum((v_1_2_flat-v3_flat)**2))
print('mean square error (V12 from lightspace with V3): ',sum((v_1_2_from_light_flat-v3_flat)**2))

cv2.waitKey(0)
cv2.destroyAllWindows()