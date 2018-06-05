import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, c):
    k = 8
    return x * np.power(k, a * c) / np.power((1 + np.power(x, 1 / c) * (np.power(k, a) - 1)), c)

def bin_count(comparagram,color):
    fit_curve = np.zeros(256)
    for i in range(256):
        indices = np.argmax(comparagram[:,i])
        fit_curve[i] = indices
    np.savetxt("result/fit_" + color + "fit.txt", fit_curve, fmt="%d")
    x_axis = np.array(range(256))
    plt.plot(x_axis[fit_curve > 0],fit_curve[fit_curve > 0], c='black', linewidth=2.0)
    plt.savefig("result/fit_"+ color + ".jpg")
    plt.clf()
    return fit_curve, x_axis

set_dir = ["8xset1/","8xset2/","8xset3/"]
compsum = 0
for i in set_dir:
    comp1 = np.loadtxt(i+'out_125_1000.txt')
    comp2 = np.loadtxt(i+'out_1000_8000.txt')
    comp3 = np.loadtxt(i+'out_8000_64000.txt')
    compsum += comp1 + comp2 + comp3



np.savetxt("compsum_mono_raw.txt",compsum,fmt="%d")

compsum[compsum>255] = 255
sum = Image.fromarray(np.uint8(compsum,mode='L'))

sum.save('compsum_mono.jpg')

compsum_loaded = np.loadtxt('compsum_mono_raw.txt')

fit,x = bin_count(np.flip(compsum_loaded,0),'mono')
print(fit)
x_data = (x+0.5) / 256.
y_data = (fit+0.5) / 256.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data,maxfev=1000)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_mono.png')
plt.clf()