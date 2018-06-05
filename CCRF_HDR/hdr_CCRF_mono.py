import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import scipy.ndimage
EPSILON=sys.float_info.epsilon
# Mono
a = 153.105
c = 0.006

f_inverse_LUT = {}

f_LUT = {}

certainty_LUT = {}

CCRF_LUT = np.zeros((256,256))

def f(q,a,c,func=None):
    if func is not None:
        return func(q)
    q_a=np.power(q,a)
    return np.power((q_a/(1+q_a)),c)

def f_inverse(f,a,c,func=None):
    if func is not None:
        return func(f)

    f_cth_root=np.power(f,1/c)
    return np.power( f_cth_root/(1-f_cth_root) ,1/a)

def conf(q,ki,a,c):
    q=np.divide(q,ki)
    q_a = np.power(q, a)
    return a*c * np.divide(np.power((1/(1+(1/q_a))),(1+c)),q_a)

def create_certainty_LUT():
    global certainty_LUT
    for i in range(256):
        q_a = np.power(f_inverse_LUT[i], a)
        certainty_LUT[f_inverse_LUT[i]] = a*c * np.divide(np.power((1/(1+(1/q_a))),(1+c)),q_a)

def create_f_inverse_LUT():
    global f_inverse_LUT
    for f in range(256):
        f_inverse_LUT[f] = f_inverse((f + 0.5) / 256., a=a, c=c)

def create_f_LUT():
    global f_LUT
    for f in range(256):
        f_LUT[f_inverse_LUT[f]] = f

def round_matrix(m):
    ret = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            ret[i][j] = int(round(m[i][j]))
    return ret


def sigmaF1(comparasum):
    sum_col = np.zeros((256,))
    cdf_col = np.zeros((256,256))
    sum_col_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_col_tmp += comparasum[j][i]
            cdf_col[j][i] = sum_col_tmp
        sum_col_tmp = 0
    np.savetxt('img/cdf_col.txt',cdf_col,fmt="%d")
    for i in range(256):
        sum_col[i] = np.sum(comparasum[:,i])
        Q1[i] = (sum_col[i]+1)*0.25
        Q3[i] = (sum_col[i]+1)*0.75

        for j in range(256):
            if (Q1[i] < cdf_col[j][i]):
                Q1[i] = j + (Q1[i]-cdf_col[j-1][i])/(cdf_col[j][i] - cdf_col[j-1][i])
                break
        for j in range(256):
            if (Q3[i] < cdf_col[j][i]):
                Q3[i] = j + (Q3[i]-cdf_col[j-1][i])/(cdf_col[j][i] - cdf_col[j-1][i])
                break
    IQR = Q3 - Q1

    #result = np.zeros((256,))
    #for i in range(256):
    #    result[i] = (IQR[i] / sum_col[i])*max(sum_col)
    #print(result)
    sigma = np.divide(IQR,1.349)
    return sigma

def sigmaF2(comparasum):
    # sigma of all rows
    sum_row = np.zeros((256,))
    cdf_row = np.zeros((256, 256))
    sum_row_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_row_tmp += comparasum[i][j]
            cdf_row[i][j] = sum_row_tmp
        sum_row_tmp = 0
    np.savetxt('img/cdf_row.txt', cdf_row, fmt="%d")
    for i in range(256):
        sum_row[i] = np.sum(comparasum[i,:])
        Q1[i] = (sum_row[i]+1)*0.25
        Q3[i] = (sum_row[i]+1)*0.75
        for j in range(256):
            if (Q1[i] < cdf_row[i][j]):
                Q1[i] = j + (Q1[i]-cdf_row[i][j-1])/(cdf_row[i][j] - cdf_row[i][j-1])
                break
        for j in range(256):
            if (Q3[i] < cdf_row[i][j]):
                Q3[i] = j + (Q3[i]-cdf_row[i][j-1])/(cdf_row[i][j] - cdf_row[i][j-1])
                break
    IQR1 = Q3 - Q1

    sigma = np.divide(IQR1,1.349)
    return sigma



def smoothsigmas(sigmaf1,sigmaf2):
    print(sigmaf1)
    print(sigmaf2)
    plt.ylim(ymax=20)
    plt.plot(sigmaf1,'ro',label='sigma(f1)-- unsmoothed')
    smooth_f1 = scipy.ndimage.gaussian_filter(sigmaf1, 5)
    plt.plot(smooth_f1,'r',label='sigma(f1)-- smoothed')

    plt.plot(sigmaf2,'bo',label='sigma(f2)-- unsmoothed')
    smooth_f2 = scipy.ndimage.gaussian_filter(sigmaf2, 5)
    plt.plot(smooth_f2,'b',label='sigma(f2)-- smoothed')
    plt.legend()
    plt.show()
    return smooth_f1,smooth_f2


def CCRF_argmin(f1,f2,sigmaf1,sigmaf2,a,c,f_inverse_LUT):
    results = np.zeros((256,))
    for i in range(256):
        q = f_inverse_LUT[i]
        q_k = q*8
        results[i] = ((((f1+0.5)/256. - f(q,a, c))**2)/((sigmaf1[f1])**2)) + \
        ((((f2+0.5)/256. - f(q_k, a, c)) ** 2) / ((sigmaf2[f2]) ** 2))
    result = f_inverse_LUT[np.argmin(results)]
    return f(result,a,c) * 256. -0.5



def create_CCRF_LUT(f_inverse_LUT,sigmaf1,sigmaf2,color,a,c):
    CCRF = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            CCRF[i][j] = CCRF_argmin(i, j, sigmaf1, sigmaf2,a,c,f_inverse_LUT)
    CCRF = round_matrix(CCRF).astype(np.uint8)
    np.savetxt('img/CCRF_'+color+'.txt', CCRF, fmt="%d")
    output = Image.fromarray(CCRF, mode='L')
    output.save('img/CCRF_'+color+'.jpg')
    return CCRF


def run_CCRF(f2q,fq):
    CCRF = np.zeros(fq.shape)
    for i in range(f2q.shape[0]):
        for j in range(f2q.shape[1]):
            X = fq[i][j]
            Y = f2q[i][j]
            CCRF[i][j] = CCRF_LUT[X][Y]
    return CCRF


create_f_inverse_LUT()
#create_f_LUT()
#create_certainty_LUT()
print(f_inverse_LUT)

# CCRF
#compsum = np.loadtxt('compsum_mono_raw.txt')
#sigmaf1 = sigmaF1(np.flip(compsum,0))
#sigmaf2 = sigmaF2(np.flip(compsum,0))
#smoothf1, smoothf2 = smoothsigmas(sigmaf1,sigmaf2)

#CCRF_LUT_B = create_CCRF_LUT(f_inverse_LUT,smoothf1,smoothf2,'mono',a = a, c = c)


def create_HDR_LUT():
    total_conf = np.zeros((256,256))
    ret = np.zeros((256,256))
    for f2q in range(256):
        for fq in range(256):
            lightspace_q = f_inverse((fq+0.5)/256.,a=a, c=c)
            lightspace_2q = f_inverse((f2q+0.5)/256.,a=a, c=c)
            cur_conf = conf(lightspace_q, ki=1,a=a, c=c) / 1
            total_conf[fq,f2q] += conf(lightspace_q, ki=1,a=a, c=c)
            ret[fq,f2q] += np.multiply(cur_conf, lightspace_q)
            cur_conf = conf(lightspace_2q, ki=2,a=a, c=c) / 2
            total_conf[fq, f2q] += conf(lightspace_2q, ki=2,a=a, c=c)
            ret[fq, f2q] += np.multiply(cur_conf, lightspace_2q)

    ret = np.divide(ret, total_conf)
    HDR_LUT =  f(ret,a=a, c=c)*256.-0.5
    HDR_LUT = round_matrix(HDR_LUT)
    return HDR_LUT





