import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, c):
    # k = 2
    #print(max(x), min(x))

    return x * np.power(8, a * c) / np.power((1 + np.power(x, 1 / c) * (np.power(8, a) - 1)), c)


def matadd(m1,m2):
    result = np.zeros(m1.shape)
    assert m1.shape == m2.shape
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            result[i][j] = m1[i][j] + m2[i][j]
    return result

def bin_count(comparagram,color):
    fit_curve = np.zeros(256)
    for i in range(256):
        indices = np.argmax(comparagram[:,i])
        #print(indices)
        fit_curve[i] = indices
    np.savetxt("result/fit_" + color + "fit.txt", fit_curve, fmt="%d")
    x_axis = np.array(range(256))
    plt.plot(x_axis[fit_curve > 0],fit_curve[fit_curve > 0], c='black', linewidth=2.0)
    plt.savefig("result/fit_"+ color + ".jpg")
    plt.clf()
    return fit_curve, x_axis

print(np.__version__)

#set_dir = ['rgbset{}/'.format]

set_dir = ["rgbset1/","rgbset2/","rgbset3/","rgbset4/","rgbset5/",\
"rgbset6/","rgbset7/","rgbset8/","rgbset9/","rgbset10/","rgbset11/",\
"rgbset12/","rgbset13/","rgbset14/","rgbset15/","rgbset16/"]
#temp = ["tmp/"]
#set_dir = temp
compsum_B = 0
compsum_G = 0
compsum_R = 0
for i in set_dir:
    comp1_B = np.loadtxt(i + 'out_15_125_B.txt')
    comp2_B = np.loadtxt(i + 'out_125_1000_B.txt')
    comp3_B = np.loadtxt(i + 'out_1000_8000_B.txt')
    comp4_B = np.loadtxt(i + 'out_8000_64000_B.txt')
    '''
    comp1_B = np.loadtxt(i + 'out_15_31_G.txt')
    comp2_B = np.loadtxt(i + 'out_31_62_G.txt')
    comp3_B = np.loadtxt(i + 'out_62_125_G.txt')
    comp4_B = np.loadtxt(i + 'out_125_250_G.txt')
    comp5_B = np.loadtxt(i + 'out_250_500_G.txt')
    comp6_B = np.loadtxt(i + 'out_500_1000_G.txt')
    comp7_B = np.loadtxt(i + 'out_1000_2000_G.txt')
    comp8_B = np.loadtxt(i + 'out_2000_4000_G.txt')
    comp9_B = np.loadtxt(i + 'out_4000_8000_G.txt')
    comp10_B = np.loadtxt(i + 'out_8000_16000_G.txt')
    comp11_B = np.loadtxt(i + 'out_16000_32000_G.txt')
    comp12_B = np.loadtxt(i + 'out_32000_64000_G.txt')
    comp13_B = np.loadtxt(i + 'out_64000_128000_G.txt')
    compsum_B += comp1_B + comp2_B + comp3_B + comp4_B + comp5_B \
                 + comp6_B + comp7_B + comp8_B + comp9_B + comp10_B + \
                comp11_B + comp12_B + comp13_B
    '''
    compsum_B += comp1_B + comp2_B + comp3_B + comp4_B

    comp1_G = np.loadtxt(i + 'out_15_125_G.txt')
    comp2_G = np.loadtxt(i + 'out_125_1000_G.txt')
    comp3_G = np.loadtxt(i + 'out_1000_8000_G.txt')
    comp4_G = np.loadtxt(i + 'out_8000_64000_G.txt')
    compsum_G += comp1_G + comp2_G + comp3_G + comp4_G
    '''
    comp1_G = np.loadtxt(i + 'out_15_31_G.txt')
    comp2_G = np.loadtxt(i + 'out_31_62_G.txt')
    comp3_G = np.loadtxt(i + 'out_62_125_G.txt')
    comp4_G = np.loadtxt(i + 'out_125_250_G.txt')
    comp5_G = np.loadtxt(i + 'out_250_500_G.txt')
    comp6_G = np.loadtxt(i + 'out_500_1000_G.txt')
    comp7_G = np.loadtxt(i + 'out_1000_2000_G.txt')
    comp8_G = np.loadtxt(i + 'out_2000_4000_G.txt')
    comp9_G = np.loadtxt(i + 'out_4000_8000_G.txt')
    comp10_G = np.loadtxt(i + 'out_8000_16000_G.txt')
    comp11_G = np.loadtxt(i + 'out_16000_32000_G.txt')
    comp12_G = np.loadtxt(i + 'out_32000_64000_G.txt')
    comp13_G = np.loadtxt(i + 'out_64000_128000_G.txt')
    compsum_G += comp1_G + comp2_G + comp3_G + comp4_G + comp5_G\
                 + comp6_G + comp7_G + comp8_G + comp9_G + comp10_G + \
                 comp11_G + comp12_G + comp13_G
    
    comp1_R = np.loadtxt(i + 'out_15_31_R.txt')
    comp2_R = np.loadtxt(i + 'out_31_62_R.txt')
    comp3_R = np.loadtxt(i + 'out_62_125_R.txt')
    comp4_R = np.loadtxt(i + 'out_125_250_R.txt')
    comp5_R = np.loadtxt(i + 'out_250_500_R.txt')
    comp6_R = np.loadtxt(i + 'out_500_1000_R.txt')
    comp7_R = np.loadtxt(i + 'out_1000_2000_R.txt')
    comp8_R = np.loadtxt(i + 'out_2000_4000_R.txt')
    comp9_R = np.loadtxt(i + 'out_4000_8000_R.txt')
    comp10_R = np.loadtxt(i + 'out_8000_16000_R.txt')
    comp11_R = np.loadtxt(i + 'out_16000_32000_R.txt')
    comp12_R = np.loadtxt(i + 'out_32000_64000_R.txt')
    comp13_R = np.loadtxt(i + 'out_64000_128000_R.txt')
    compsum_R += comp1_R + comp2_R + comp3_R + comp4_R + comp5_R \
                 + comp6_R + comp7_R + comp8_R + comp9_R + comp10_R + \
                 comp11_R + comp12_R + comp13_R
    '''
    comp1_R = np.loadtxt(i + 'out_15_125_R.txt')
    comp2_R = np.loadtxt(i + 'out_125_1000_R.txt')
    comp3_R = np.loadtxt(i + 'out_1000_8000_R.txt')
    comp4_R = np.loadtxt(i + 'out_8000_64000_R.txt')
    compsum_R += comp1_R + comp2_R + comp3_R + comp4_R

np.savetxt("compsum_B_raw.txt",compsum_B,fmt="%d")
np.savetxt("compsum_G_raw.txt",compsum_G,fmt="%d")
np.savetxt("compsum_R_raw.txt",compsum_R,fmt="%d")

compsum_B[compsum_B>255] = 255
compsum_G[compsum_G>255] = 255
compsum_R[compsum_R>255] = 255
sum_B = Image.fromarray(np.uint8(compsum_B,mode='L'))
sum_G = Image.fromarray(np.uint8(compsum_G,mode='L'))
sum_R = Image.fromarray(np.uint8(compsum_R,mode='L'))

sum_B.save('compsum_B.jpg')
sum_G.save('compsum_G.jpg')
sum_R.save('compsum_R.jpg')

compsum_loaded_B = np.loadtxt('compsum_B_raw.txt')
compsum_loaded_G = np.loadtxt('compsum_G_raw.txt')
compsum_loaded_R = np.loadtxt('compsum_R_raw.txt')



fit_B,x_B = bin_count(np.flip(compsum_loaded_B,0),'B')
x_data = (x_B+0.5) / 256.
y_data = (fit_B+0.5) / 256.
#y_data = y_data.astype(np.float32)
#x_data = x_data.astype(np.float32)
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_B.png')
plt.clf()


fit_R,x_R = bin_count(np.flip(compsum_loaded_R,0),'R')
x_data = (x_R+0.5) / 256.
y_data = (fit_R+0.5) / 256.
#y_data = y_data.astype(np.float32)
#x_data = x_data.astype(np.float32)
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_R.png')
plt.clf()


fit_G,x_G = bin_count(np.flip(compsum_loaded_G,0),'G')
x_data = (x_G+0.5) / 256.
y_data = (fit_G+0.5) / 256.
#y_data = y_data.astype(np.float32)
#x_data = x_data.astype(np.float32)
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_G.png')
plt.clf()