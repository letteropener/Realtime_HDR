import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
EPSILON=sys.float_info.epsilon

# R channel a = 1.638  c = 0.533
# G channel a = 1.677  c = 0.527
# B channel a = 1.732  c = 0.506
# Avg a = 1.682 c = 0.522

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

def conf(q,ki=1,a=1.682,c=0.522,func=None):
    if a!=1.682 or c!=0.522:
        return func(q)
    q=np.divide(q,ki)
    certainty = np.zeros(q.shape)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            for k in range(q.shape[2]):
                if q[i][j][k] > 0.0:
                    q1_682 = np.power(q[i][j][k], a)
                    certainty[i][j][k] = 0.878 * np.divide(np.power((1/(1+(1/q1_682))),(1+c)),q1_682)
    return certainty

def get_certainty(q,a=1.682,c=0.522):
    certainty = np.zeros(q.shape,)
    for i in range(q.shape[0]):
        if q[i] > 0.0:
            q1_682 = np.power(q[i], a)
            certainty[i] = 0.878 * np.divide(np.power((1/(1+(1/q1_682))),(1+c)),q1_682)

    return certainty


def get_q_hat(v,ki=1):
    q = f_inverse(v)
    q_hat = np.divide(q,ki)
    return q_hat

def preprocess(image):
    
    #normalize pixels to [0,1] range
    image=image/255.
    image[image>=1]=1-EPSILON
    return image

def hdr_merge(images,steps):
    
    total_conf=np.zeros(images[0].shape)
    ret = np.zeros(images[0].shape)

    for image,ki in zip(images,steps):
        lightspace_q=f_inverse(image)
        cur_conf=conf(lightspace_q,ki=ki)/ ki
        total_conf+=conf(lightspace_q,ki=ki)
        
        ret+=np.multiply(cur_conf,lightspace_q)

    ret=np.divide(ret,total_conf)
    return f(ret)


very_high=cv2.imread("img/half_a_sec.jpg")
very_high=preprocess(very_high)
high=cv2.imread("img/4th_a_sec.jpg")
high=preprocess(high)
mid=cv2.imread("img/8th_a_sec.jpg")
mid=preprocess(mid)
low=cv2.imread("img/16th_a_sec.jpg")
low=preprocess(low)
very_low=cv2.imread("img/32th_a_sec.jpg")
very_low=preprocess(very_low)
num_pix=very_high.shape[0]*very_high.shape[1]


very_high_flat = very_high.reshape(num_pix,3)
high_flat = high.reshape(num_pix,3)
mid_flat = mid.reshape(num_pix,3)
low_flat = low.reshape(num_pix,3)
very_low_flat = very_low.reshape(num_pix,3)

very_high_b=very_high_flat[:,0]
very_high_g=very_high_flat[:,1]
very_high_r=very_high_flat[:,2]
q_4hat_b = get_q_hat(very_high_b)
certainty_4hat_b = get_certainty(np.sort(q_4hat_b))
plt.plot(np.sort(q_4hat_b),np.sort(very_high_b),c='red',label='response curve')
plt.plot(np.sort(q_4hat_b),certainty_4hat_b,c='blue',label='certainty func')

plt.title('q_4 B channel')
plt.legend()
plt.xscale('log')
plt.xlim([0.01,100])
plt.savefig('q_4 B channel')
plt.clf()

high_b=high_flat[:,0]
high_g=high_flat[:,1]
high_r=high_flat[:,2]
q_2hat_b = get_q_hat(high_b)
certainty_2hat_b = get_certainty(np.sort(q_2hat_b))
plt.plot(np.sort(q_2hat_b),np.sort(high_b),c='green',label='response curve')
plt.plot(np.sort(q_2hat_b),certainty_2hat_b,c='blue',label='certainty func')

plt.title('q_2 B channel')
plt.legend()
plt.xscale('log')
plt.xlim([0.01,100])
plt.savefig('q_2 B channel')
plt.clf()

mid_b=mid_flat[:,0]
mid_g=mid_flat[:,1]
mid_r=mid_flat[:,2]
q_1hat_b = get_q_hat(mid_b)
certainty_1hat_b = get_certainty(np.sort(q_1hat_b))
plt.plot(np.sort(q_1hat_b),np.sort(mid_b),c='blue',label='response curve')
plt.plot(np.sort(q_1hat_b),certainty_1hat_b,c='red',label='certainty func')

plt.title('q_1 B channel')
plt.legend()
plt.xscale('log')
plt.xlim([0.01,100])
plt.savefig('q_1 B channel')
plt.clf()

low_b=low_flat[:,0]
low_g=low_flat[:,1]
low_r=low_flat[:,2]

q_0_5hat_b = get_q_hat(low_b)
certainty_0_5hat_b = get_certainty(np.sort(q_0_5hat_b))
plt.plot(np.sort(q_0_5hat_b),np.sort(low_b),c='brown',label='response curve')
plt.plot(np.sort(q_0_5hat_b),certainty_0_5hat_b,c='blue',label='certainty func')

plt.title('q_0.5 B channel')
plt.legend()
plt.xscale('log')
plt.xlim([0.01,100])
plt.savefig('q 0_5 B channel')
plt.clf()

very_low_b=very_low_flat[:,0]
very_low_g=very_low_flat[:,1]
very_low_r=very_low_flat[:,2]
q_0_25hat_b = get_q_hat(very_low_b)
certainty_0_25hat_b = get_certainty(np.sort(q_0_25hat_b))
plt.plot(np.sort(q_0_25hat_b),np.sort(very_low_b),c='pink',label='response curve')
plt.plot(np.sort(q_0_25hat_b),certainty_0_25hat_b,c='blue',label='certainty func')

plt.title('q_0.25 B channel')
plt.legend()
plt.xscale('log')
plt.xlim([0.01,100])
plt.savefig('q 0_25 B channel')
plt.clf()


hdr=hdr_merge([very_high, high, mid, low, very_low],[4,2,1,0.5,0.25])*255

hdr=hdr.astype(np.int32)
#print(hdr.shape)
cv2.imwrite("out.jpg",hdr)
out=cv2.imread("out.jpg")
cv2.imshow("hdr",out)
cv2.imshow("very high",very_high)
cv2.imshow("high",high)
cv2.imshow('mid',mid)
cv2.imshow('low',low)
cv2.imshow('very low',very_low)


cv2.waitKey(0)
cv2.destroyAllWindows()