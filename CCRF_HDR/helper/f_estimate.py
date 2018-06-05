from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import curve_fit



def generate_comparagram(fq_flat,f2q_flat,filepair,color):
    comparagram = np.zeros((256, 256))
    for x, y in zip(fq_flat, f2q_flat):
        comparagram[x][y] += 1
    # print(comparagram)
    X_axis = []
    Y_axis = []
    for i in range(256):
        for j in range(256):
            if comparagram[i][j] > 0:
                X_axis.append(float(i))
                Y_axis.append(float(j))
    plt.title(filepair+'_'+color + ' channel comparagram')
    plt.scatter(X_axis, Y_axis,c=color)
    plt.savefig(filepair+'_'+color + ' channel comparagram')
    np.savetxt(filepair+'_'+color + " channel histogram.txt", comparagram, fmt="%d")
    plt.clf()


    return comparagram


def bin_count(comparagram,filepair,color):
    fit_curve = np.zeros(256)
    for i in range(256):
        indices = np.argmax(comparagram[i])
        #print(indices)
        fit_curve[i] = indices.mean()
    np.savetxt(filepair+'_'+ color+"fit.txt", fit_curve, fmt="%d")

    x_axis = np.array(range(256))
    plt.plot(x_axis[fit_curve > 0], fit_curve[fit_curve > 0],c='black',linewidth=2.0)
    plt.savefig(filepair+'_'+ color+"fit.png")
    plt.clf()
    return fit_curve,x_axis

def func(x, a, c):
    # k = 2
    return x * np.power(2, a * c) / np.power((1 + np.power(x, 1 / c) * (np.power(2, a) - 1)), c)


def grad_descent(x_axis,fit_curve,filepair,color):
    print("tensor flow starts")

    x = (x_axis[fit_curve > 0] / 255.).astype(np.float32)
    # print(x.dtype)
    y_data = (fit_curve[fit_curve > 0] / 255.).astype(np.float32)
    # print(y_data.dtype)

    a = tf.Variable(1.)
    c = tf.Variable(1.)
    k = tf.constant(2.0)
    y = x * tf.pow(k, a * c) / tf.pow((1 + tf.pow(x, 1 / c) * (tf.pow(k, a) - 1)), c)  # P145 (4.58)

    loss = tf.reduce_mean(tf.square(np.array(y_data) - y))
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(2000001):
            sess.run(train)
            if step % 200 == 0:
                print(step,filepair+'_'+color, sess.run([a, c]), 'loss= ', sess.run(loss))


def compsum(compsum,color):
    X_axis = []
    Y_axis = []
    for i in range(256):
        for j in range(256):
            if compsum[i][j] > 0:
                X_axis.append(float(i))
                Y_axis.append(float(j))
    plt.title(color+' channel comparagramSum')
    plt.scatter(X_axis, Y_axis,c=color)
    plt.savefig(color+' channel comparagramSum')
    plt.clf()



#f16q=cv2.imread("img/half_a_sec.jpg")
#f8q=cv2.imread("img/4th_a_sec.jpg")
#f4q=cv2.imread("img/8th_a_sec.jpg")
#f2q=cv2.imread("img/16th_a_sec.jpg")
#fq=cv2.imread("img/32th_a_sec.jpg")

f8q=cv2.imread("img/8000.jpg")
f4q=cv2.imread("img/4000.jpg")
f2q=cv2.imread("img/2000.jpg")
fq=cv2.imread("img/1000.jpg")
assert fq.shape==f2q.shape
num_pix=fq.shape[0]*fq.shape[1]
#f16q_flat=f16q.reshape(num_pix,3)
f8q_flat=f8q.reshape(num_pix,3)
f4q_flat=f4q.reshape(num_pix,3)
f2q_flat=f2q.reshape(num_pix,3)
fq_flat=fq.reshape(num_pix,3)
#opencv opens imgs in BGR channels
fq_flat_b=fq_flat[:,0]
fq_flat_g=fq_flat[:,1]
fq_flat_r=fq_flat[:,2]
f2q_flat_b=f2q_flat[:,0]
f2q_flat_g=f2q_flat[:,1]
f2q_flat_r=f2q_flat[:,2]
f4q_flat_b=f4q_flat[:,0]
f4q_flat_g=f4q_flat[:,1]
f4q_flat_r=f4q_flat[:,2]
f8q_flat_b=f8q_flat[:,0]
f8q_flat_g=f8q_flat[:,1]
f8q_flat_r=f8q_flat[:,2]
#f16q_flat_b=f16q_flat[:,0]
#f16q_flat_g=f16q_flat[:,1]
#f16q_flat_r=f16q_flat[:,2]

comparagram1_b = generate_comparagram(fq_flat_b,f2q_flat_b,'32th_16th','B')
x_axis,fit_curve = bin_count(comparagram1_b,'32th_16th','B')
comparagram1_g = generate_comparagram(fq_flat_g,f2q_flat_g,'32th_16th','G')
x_axis,fit_curve = bin_count(comparagram1_g,'32th_16th','G')
comparagram1_r = generate_comparagram(fq_flat_r,f2q_flat_r,'32th_16th','R')
x_axis,fit_curve = bin_count(comparagram1_r,'32th_16th','R')

comparagram2_b = generate_comparagram(f2q_flat_b,f4q_flat_b,'16th_8th','B')
x_axis,fit_curve = bin_count(comparagram2_b,'16th_8th','B')
comparagram2_g = generate_comparagram(f2q_flat_g,f4q_flat_g,'16th_8th','G')
x_axis,fit_curve = bin_count(comparagram2_g,'16th_8th','G')
comparagram2_r = generate_comparagram(f2q_flat_r,f4q_flat_r,'16th_8th','R')
x_axis,fit_curve = bin_count(comparagram2_r,'16th_8th','R')

comparagram3_b = generate_comparagram(f4q_flat_b,f8q_flat_b,'8th_4th','B')
x_axis,fit_curve = bin_count(comparagram3_b,'8th_4th','B')
comparagram3_g = generate_comparagram(f4q_flat_g,f8q_flat_g,'8th_4th','G')
x_axis,fit_curve = bin_count(comparagram3_g,'8th_4th','G')
comparagram3_r = generate_comparagram(f4q_flat_r,f8q_flat_r,'8th_4th','R')
x_axis,fit_curve = bin_count(comparagram3_r,'8th_4th','R')

#comparagram4_b = generate_comparagram(f8q_flat_b,f16q_flat_b,'half_4th','B')
#x_axis,fit_curve = bin_count(comparagram4_b,'half_4th','B')
#comparagram4_g = generate_comparagram(f8q_flat_g,f16q_flat_g,'half_4th','G')
#x_axis,fit_curve = bin_count(comparagram4_g,'half_4th','G')
#comparagram4_r = generate_comparagram(f8q_flat_r,f16q_flat_r,'half_4th','R')
#x_axis,fit_curve = bin_count(comparagram4_r,'half_4th','R')



#compsum_b = comparagram1_b + comparagram2_b + comparagram3_b + comparagram4_b
compsum_b = comparagram1_b + comparagram2_b + comparagram3_b
compsum(compsum_b,'B')
x_axis_B, fit_curve_B = bin_count(compsum_b,"All",'B')
#grad_descent(x_axis_B,fit_curve_B,"All",'B')
y_data = x_axis_B[fit_curve_B > 0] / 255.
x_data = fit_curve_B[fit_curve_B > 0] / 255.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('q')
plt.ylabel('f(q)')
popt_ac = [popt[0],popt[0],popt[1]]
plt.title('f(q) = (q^%5.3f / (1+q^%5.3f))^%5.3f' % tuple(popt_ac))
plt.legend()
plt.savefig('fit_curve_B.png')
plt.clf()

#compsum_g = comparagram1_g + comparagram2_g + comparagram3_g + comparagram4_g
compsum_g = comparagram1_g + comparagram2_g + comparagram3_g
compsum(compsum_g,'G')
x_axis_G, fit_curve_G = bin_count(compsum_g,"All",'G')
#grad_descent(x_axis_G,fit_curve_G,"All",'G')
y_data = x_axis_G[fit_curve_G > 0] / 255.
x_data = fit_curve_G[fit_curve_G > 0] / 255.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('q')
plt.ylabel('f(q)')
popt_ac = [popt[0],popt[0],popt[1]]
plt.title('f(q) = (q^%5.3f / (1+q^%5.3f))^%5.3f' % tuple(popt_ac))
plt.legend()
plt.savefig('fit_curve_G.png')
plt.clf()

#compsum_r = comparagram1_r + comparagram2_r + comparagram3_r + comparagram4_r
compsum_r = comparagram1_r + comparagram2_r + comparagram3_r
compsum(compsum_r,'R')
x_axis_R, fit_curve_R = bin_count(compsum_r,"All",'R')
#grad_descent(x_axis_R,fit_curve_R,"All",'R')
y_data = x_axis_R[fit_curve_R > 0] / 255.
x_data = fit_curve_R[fit_curve_R > 0] / 255.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('q')
plt.ylabel('f(q)')
popt_ac = [popt[0],popt[0],popt[1]]
plt.title('f(q) = (q^%5.3f / (1+q^%5.3f))^%5.3f' % tuple(popt_ac))
plt.legend()
plt.savefig('fit_curve_R.png')
plt.clf()
