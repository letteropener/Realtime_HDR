import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from config import *
import datetime
set_id=None
sum_id=None

if os.path.isfile(flag_file) is False:
    raise ValueError("No new photo set detected. Abort")

if os.path.isdir( comparasum_dir) is False:
    os.mkdir(comparasum_dir)

try:
    with open(os.path.join(data_dir,marker_filename)) as fin:
        #acquire id of latest img set
        set_id=fin.readline()
    print("Debug - latest set is:{}".format(set_id))
except:
    raise OSError("Cannot find file: data/latest_set.txt")

#find latest comparasum
try:
    with open(os.path.join(comparasum_dir,comparasum_tracker)) as fin:
        sum_id=fin.readline()
except:
    sum_id=None


full_set_dir=os.path.join(data_dir,set_id)


    
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



print("Debug - sum_id: {}".format(sum_id))

#set_dir = ["photo/" + folders for folders in os.listdir("photo") if os.path.isdir("photo/"+folders) == True]
k = 2
new_sum_id=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
new_sum_dir=os.path.join(comparasum_dir ,new_sum_id)
os.makedirs(new_sum_dir)
for index in range(1,4):
    ki = 2**index
    if sum_id is None:
        sum_r=np.zeros((256,256))
        sum_g=np.zeros((256,256))
        sum_b=np.zeros((256,256))

    else:
        #load files
        full_dir=os.path.join(comparasum_dir,set_id)
        try:
            sum_r=np.loadtxt(os.path.join(full_dir,"comparasum_R_raw_{}.txt".format(ki)))
            sum_g=np.loadtxt(os.path.join(full_dir,"comparasum_G_raw_{}.txt".format(ki)))
            sum_B=np.loadtxt(os.path.join(full_dir,"comparasum_B_raw_{}.txt".format(ki)))
        except:
            sum_r=np.zeros((256,256))
            sum_g=np.zeros((256,256))
            sum_b=np.zeros((256,256))
   

    #load comparagram from latest set 
    file_list_g = ['out_{}_{}_G.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]
    file_list_b = ['out_{}_{}_B.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]
    file_list_r = ['out_{}_{}_R.txt'.format(exposures[i],exposures[i+index]) for i in range(len(exposures)-index)]

    for file_r,file_g,file_b in zip(file_list_r,file_list_g,file_list_b):
        sum_r+=np.loadtxt(os.path.join(full_set_dir,file_r))
        sum_g+=np.loadtxt(os.path.join(full_set_dir,file_g))
        sum_b+=np.loadtxt(os.path.join(full_set_dir,file_b))

        np.savetxt(os.path.join(new_sum_dir,"comparasum_R_raw_{}.txt".format(ki)),sum_r,fmt="%d")
        np.savetxt(os.path.join(new_sum_dir,"comparasum_G_raw_{}.txt".format(ki)),sum_g,fmt="%d")
        np.savetxt(os.path.join(new_sum_dir,"comparasum_B_raw_{}.txt".format(ki)),sum_b,fmt="%d")

        #save to jpg
        #clip values greater than 255 to 255
       
        sum_r[sum_r>255]=255
        sum_g[sum_g>255]=255
        sum_b[sum_b>255]=255
        img_R = Image.fromarray(np.uint8(sum_r,mode='L'))
        img_G = Image.fromarray(np.uint8(sum_g,mode='L'))
        img_B = Image.fromarray(np.uint8(sum_b,mode='L'))

        img_R.save(os.path.join(new_sum_dir,"comparasum_R_raw_{}.jpg".format(ki)))
        img_G.save(os.path.join(new_sum_dir,"comparasum_G_raw_{}.jpg".format(ki)))
        img_B.save(os.path.join(new_sum_dir,"comparasum_B_raw_{}.jpg".format(ki)))


#update tracker file

with open(os.path.join(comparasum_dir,comparasum_tracker),'w') as fout:
    fout.write(new_sum_id)

try:
    os.remove(flag_file)
except:
    print("Cannot locate flag file {}".format(flag_file))
    raise





'''

    for i in set_dir:
        print("current folder:{}/{},index: {}/{}".format(count,len(set_dir),index,4))
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

'''
'''
#Curve fit for RGB channels and save results
        fit_B,x_B = bin_count(np.flip(compsum_loaded_B,0),'B')
        x_data = (x_B+0.5) / 256.
        y_data = (fit_B+0.5) / 256.
        #y_data = y_data.astype(np.float32)
        #x_data = x_data.astype(np.float32)
        plt.plot(x_data,y_data)
        popt, pcov = curve_fit(func, x_data, y_data,maxfev=10000)
        with open("fparam_{}_B.txt".format(k**index),'w') as f:
            f.write("{},{}".format(*popt))
        plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
        plt.legend()
        plt.savefig('result/fit_curve_B_{}.png'.format(k**index))
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
        plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
        plt.legend()
        plt.savefig('result/fit_curve_R_{}.png'.format(k**index))
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
        plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
        plt.legend()
        plt.savefig('result/fit_curve_G_{}.png'.format(k**index))
        plt.clf()
'''