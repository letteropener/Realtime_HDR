import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pickle
from math import *
base = 16
bias_base = 10
bias = [log(bias_base+i,3) for i in range(1,540000,18000)]
exposure_range = 11
def func(x, a, c):
    # k = 2
    #print(max(x), min(x))

    return x * np.power(ki, a * c) / np.power((1 + np.power(x, 1 / c) * (np.power(ki, a) - 1)), c)


def matadd(m1,m2):
    result = np.zeros(m1.shape)
    assert m1.shape == m2.shape
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            result[i][j] = m1[i][j] + m2[i][j]
    return result

def bin_count(comparagram,color):
    x_data = np.zeros(int(np.sum(comparagram)))
    y_data = np.zeros(int(np.sum(comparagram)))
    data_pos_flag = 0
    for i in range(256):
        for j in range(256):
            if comparagram[i][j] != 0:
                for num in range(int(comparagram[i][j])):
                    y_data[data_pos_flag] = i
                    x_data[data_pos_flag] = j
                    data_pos_flag += 1
    print("total :",np.sum(int(np.sum(comparagram))))
    print("went through :",data_pos_flag)
    #np.savetxt("result/fit_" + color + "fit.txt", fit_curve, fmt="%d")
    #x_axis = np.array(range(256))
    #plt.plot(x_axis[fit_curve > 0],fit_curve[fit_curve > 0], c='black', linewidth=2.0)
    #plt.savefig("result/fit_"+ color + ".jpg")
    #plt.clf()
    return y_data, x_data

print(np.__version__)

#set_dir = ["rgbset1/","rgbset2/","rgbset3/","rgbset4/","rgbset5/",\
#"rgbset6/","rgbset7/","rgbset8/","rgbset9/","rgbset10/","rgbset11/"]

set_dir = ["photo/" + folders for folders in os.listdir("photo") if os.path.isdir("photo/"+folders) == True]
k = 2
for index in range(1,4):
    compsum_B = np.zeros((256,256),dtype="float64")
    compsum_G = np.zeros((256,256),dtype="float64")
    compsum_R = np.zeros((256,256),dtype="float64")
    for j in range(len(bias)): #loop for bias
        exposures = [(base+bias[j])*k**i for i in range(exposure_range)]
        
        ki = 2**index
        file_list_g = ['out_{}_{}_G.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]
        file_list_b = ['out_{}_{}_B.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]
        file_list_r = ['out_{}_{}_R.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]
        #temp = ["tmp/"]
        #set_dir = temp

        count = 1
        for i in set_dir:
            print("current folder:{}/{},index: {}/{}, bias: {}/{}".format(count,len(set_dir),index,3,j+1,len(bias)))
            for fname_b in file_list_b:
                compsum_B += np.loadtxt(os.path.join(i,fname_b))
            for fname_g in file_list_g:
                compsum_G += np.loadtxt(os.path.join(i,fname_g))
            for fname_r in file_list_r:
                compsum_R += np.loadtxt(os.path.join(i,fname_r))
            count+=1
        np.savetxt("compsum_B_raw_{}.txt".format(k**index),compsum_B,fmt="%d")
        np.savetxt("compsum_G_raw_{}.txt".format(k**index),compsum_G,fmt="%d")
        np.savetxt("compsum_R_raw_{}.txt".format(k**index),compsum_R,fmt="%d")

        compsum_B[compsum_B>255] = 255
        compsum_G[compsum_G>255] = 255
        compsum_R[compsum_R>255] = 255
        sum_B = Image.fromarray(np.uint8(compsum_B,mode='L'))
        sum_G = Image.fromarray(np.uint8(compsum_G,mode='L'))
        sum_R = Image.fromarray(np.uint8(compsum_R,mode='L'))

        sum_B.save('compsum_B_{}.jpg'.format(k**index))
        sum_G.save('compsum_G_{}.jpg'.format(k**index))
        sum_R.save('compsum_R_{}.jpg'.format(k**index))

    compsum_loaded_B = np.loadtxt('compsum_B_raw_{}.txt'.format(k**index))
    compsum_loaded_G = np.loadtxt('compsum_G_raw_{}.txt'.format(k**index))
    compsum_loaded_R = np.loadtxt('compsum_R_raw_{}.txt'.format(k**index))


    plot_x_data = np.linspace(0.0,1.0,num=256)
    fit_B,x_B = bin_count(np.flip(compsum_loaded_B,0),'B')
    x_data = (x_B+0.5) / 256.
    y_data = (fit_B+0.5) / 256.
    #y_data = y_data.astype(np.float32)
    #x_data = x_data.astype(np.float32)
    #print(ki)
    popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
    with open("fparam_{}_B.txt".format(k**index),'w') as f:
        f.write("{},{}".format(*popt))
    plt.plot(x_data,y_data)
    #print(*popt)
    plt.plot(plot_x_data, func(plot_x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()
    #print("Saving B k=", k**index)
    plt.savefig('fit_curve_B_{}.png'.format(k**index))
    plt.clf()


    fit_R,x_R = bin_count(np.flip(compsum_loaded_R,0),'R')
    x_data = (x_R+0.5) / 256.
    y_data = (fit_R+0.5) / 256.
    #y_data = y_data.astype(np.float32)
    #x_data = x_data.astype(np.float32)
    plt.plot(x_data,y_data)
    popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
    with open("fparam_{}_R.txt".format(k**index),'w') as f:
        f.write("{},{}".format(*popt))
    plt.plot(plot_x_data, func(plot_x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()
    plt.savefig('fit_curve_R_{}.png'.format(k**index))
    plt.clf()


    fit_G,x_G = bin_count(np.flip(compsum_loaded_G,0),'G')
    x_data = (x_G+0.5) / 256.
    y_data = (fit_G+0.5) / 256.
    #y_data = y_data.astype(np.float32)
    #x_data = x_data.astype(np.float32)
    plt.plot(x_data,y_data)
    popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
    with open("fparam_{}_G.txt".format(k**index),'w') as f:
        f.write("{},{}".format(*popt))
    plt.plot(plot_x_data, func(plot_x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()
    plt.savefig('fit_curve_G_{}.png'.format(k**index))
    plt.clf()