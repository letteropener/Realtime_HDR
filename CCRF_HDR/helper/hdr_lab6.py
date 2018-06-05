import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
EPSILON=sys.float_info.epsilon

# R channel a = 30.451  c = 0.024
# G channel a = 25.811  c = 0.028
# B channel a = 20.731  c = 0.031
# Avg a = 25.664 c = 0.0277

def f(q, a=25.664, c=0.0277, func=None):
    if func is not None:
        return func(q)
    q_a = np.power(q, a)
    return np.power((q_a / (1 + q_a)), c)


def f_inverse(f, a=25.664, c=0.0277, func=None):
    if func is not None:
        return func(f)

    f_cth_root = np.power(f, 1 / c)
    return np.power(f_cth_root / (1 - f_cth_root), 1 / a)


def conf(q, ki=1, a=25.664, c=0.0277):
    q = np.divide(q, ki)
    certainty = np.zeros(q.shape)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if q[i][j] > 0.0:
                q1_682 = np.power(q[i][j], a)
                certainty[i][j] = a*c * np.divide(np.power((1 / (1 + (1 / q1_682))), (1 + c)), q1_682)
    return certainty


def get_certainty(q, a=25.664, c=0.0277):
    certainty = np.zeros(q.shape, )
    for i in range(q.shape[0]):
        if q[i] > 0.0:
            q1_682 = np.power(q[i], a)
            certainty[i] = a*c * np.divide(np.power((1 / (1 + (1 / q1_682))), (1 + c)), q1_682)

    return certainty


def get_q_hat(v, ki=1):
    q = f_inverse(v)
    q_hat = np.divide(q, ki)
    return q_hat


def preprocess(image):
    # normalize pixels to [0,1] range
    image = (image+0.5) / 256.
    return image.astype(float)


def hdr_merge(images, steps):
    total_conf_R = np.zeros(images[0][:,:,2].shape)
    ret_R = np.zeros(images[0][:,:,2].shape)
    print(ret_R.shape)
    total_conf_G = np.zeros(images[0][:,:,1].shape)
    ret_G = np.zeros(images[0][:,:,1].shape)
    print(ret_G.shape)
    total_conf_B = np.zeros(images[0][:,:,0].shape)
    ret_B = np.zeros(images[0][:,:,0].shape)
    print(ret_B.shape)
    ret = np.zeros(images[0].shape)

    for image, ki in zip(images, steps):
        lightspace_q_R = f_inverse(image[:,:,2],a = 30.451 , c = 0.024)
        cur_conf_R = conf(lightspace_q_R, ki=ki,a = 30.451 , c = 0.024) / ki
        total_conf_R += conf(lightspace_q_R,a = 30.451 , c = 0.024, ki=ki)
        ret_R += np.multiply(cur_conf_R, lightspace_q_R)

        lightspace_q_G = f_inverse(image[:,:, 1], a = 25.811,  c = 0.028)
        cur_conf_G = conf(lightspace_q_G, ki=ki, a = 25.811,  c = 0.028) / ki
        total_conf_G += conf(lightspace_q_G, a = 25.811,  c = 0.028, ki=ki)
        ret_G += np.multiply(cur_conf_G, lightspace_q_G)

        lightspace_q_B = f_inverse(image[:,:, 0], a = 20.731 , c = 0.031)
        cur_conf_B = conf(lightspace_q_B, ki=ki, a = 20.731,  c = 0.031) / ki
        total_conf_B += conf(lightspace_q_B, a = 20.731,  c = 0.031, ki=ki)
        ret_B += np.multiply(cur_conf_B, lightspace_q_B)

    ret_R = np.divide(ret_R, total_conf_R)
    ret_G = np.divide(ret_G, total_conf_G)
    ret_B = np.divide(ret_B, total_conf_B)
    ret[:,:,0] = ret_B
    ret[:,:,1] = ret_G
    ret[:,:,2] = ret_R
    return ret


high=cv2.imread("img/1_s.jpg")
high=preprocess(high)
low=cv2.imread("img/2_s.jpg")
low=preprocess(low)

hdr=hdr_merge([high,low],[2,1])*255
cv2.imwrite("img/out.jpg",hdr)