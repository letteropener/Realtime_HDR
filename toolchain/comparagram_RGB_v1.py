from PIL import Image
import numpy as np
import cv2
import os
from math import *
photoBaseDir = 'photo'
base = 16
bias_base = 10
bias = [log(bias_base+i,3) for i in range(1,540000,18000)]
exposure_range = 11

# place 2 pictures in the dir
# 1st one with high exposure, 2nd one with low exposure
# differ by k (k = 2 in our case)
def calc_histgram(f2q,fq,filename):
    comparagram_B = np.zeros((256, 256))
    comparagram_G = np.zeros((256, 256))
    comparagram_R = np.zeros((256, 256))

    # B channel
    for x, y in zip(f2q[:,:,0].reshape(f2q.shape[0]*f2q.shape[1],), fq[:,:,0].reshape(fq.shape[0]*fq.shape[1],)):
        comparagram_B[x][y] += 1
    # B channel
    for x, y in zip(f2q[:, :, 1].reshape(f2q.shape[0]*f2q.shape[1],), fq[:, :, 1].reshape(fq.shape[0]*fq.shape[1],)):
        comparagram_G[x][y] += 1
    # B channel
    for x, y in zip(f2q[:, :, 2].reshape(f2q.shape[0]*f2q.shape[1],), fq[:, :, 2].reshape(fq.shape[0]*fq.shape[1],)):
        comparagram_R[x][y] += 1

    np.savetxt(filename + "_B.txt", np.flip(comparagram_B, 0), fmt="%d")
    np.savetxt(filename + "_G.txt", np.flip(comparagram_G, 0), fmt="%d")
    np.savetxt(filename + "_R.txt", np.flip(comparagram_R, 0), fmt="%d")
    return comparagram_B,comparagram_G,comparagram_R

photo = os.listdir(photoBaseDir)
k = 2
count = 1
for item in photo:
    if any(image_file_name.endswith(".txt") for image_file_name in os.listdir(os.path.join(photoBaseDir,item))):
        continue
    #imgFiles =["15.625.jpg","31.25.jpg","62.5.jpg","125.jpg","250.jpg","500.jpg","1000.jpg","2000.jpg",
    #          "4000.jpg","8000.jpg","16000.jpg","32000.jpg","64000.jpg","128000.jpg"
    #          ]
    for index in range(1,4): #loop for k
        #print("imgFiles = {}".format(imgFiles))
        for j in range(len(bias)): #loop for bias
            imgFiles = ["{}.jpg".format((base+bias[j])*k**i) for i in range(exposure_range)]
            #imageFiles = os.listdir(os.path.join(photoBaseDir, item))
            print("current folder:{}/{},index: {}/{}, bias: {}/{}".format(count,len(photo),index,3,j+1,len(bias)))
            img = ["{}/{}/{}".format(photoBaseDir,item,fname) for fname in imgFiles]
            #print(img)
            for i in range(len(img)-index):
                fq_element = img[i]
                f2q_element = img[i+index]
                print("fq = ",fq_element)
                print("f2q = ", f2q_element)
                fq = cv2.imread(fq_element)
                #print(fq_element)
                #print(f2q_element)
                f2q = cv2.imread(f2q_element)
                fq_fname=imgFiles[i]
                f2q_fname=imgFiles[i+index]
                B,G,R = calc_histgram(f2q,fq,photoBaseDir + "/"+ item+"/out_"+fq_fname.split('.jpg')[0]+ "_"+ f2q_fname.split('.jpg')[0])
                B[B>255] = 255
                G[G>255] = 255
                R[R>255] = 255
                img_B = Image.fromarray(np.flip(B.astype(np.uint8), 0), mode='L')
                img_G = Image.fromarray(np.flip(G.astype(np.uint8), 0), mode='L')
                img_R = Image.fromarray(np.flip(R.astype(np.uint8), 0), mode='L')
                #np.savetxt("mono_after.txt",comparagram,fmt="%d")
                img_B.save(photoBaseDir + "/"+ item+"/out_" + fq_fname.split('.jpg')[0] + "_" + f2q_fname.split('.jpg')[0] + "_B.jpg")
                img_G.save(photoBaseDir + "/" +item+"/out_" + fq_fname.split('.jpg')[0] + "_" + f2q_fname.split('.jpg')[0] + "_G.jpg")
                img_R.save(photoBaseDir + "/" +item+"/out_" + fq_fname.split('.jpg')[0] + "_" + f2q_fname.split('.jpg')[0] + "_R.jpg")
    count += 1