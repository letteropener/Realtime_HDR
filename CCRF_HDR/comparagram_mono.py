from PIL import Image
import numpy as np
import cv2


# place 2 pictures in the dir
# 1st one with high exposure, 2nd one with low exposure
# differ by k (k = 2 in our case)
def calc_histgram(f2q,fq,filename):
    comparagram = np.zeros((256, 256))
    for x, y in zip(f2q, fq):
        comparagram[x][y] += 1
    np.savetxt(filename,np.flip(comparagram,0),fmt="%d")
    return comparagram


photos = ["125.jpg","1000.jpg","8000.jpg","64000.jpg"]
for i in range(len(photos)-1):
        fq_element = photos[i]
        f2q_element = photos[i+1]
        fq = cv2.imread(fq_element,cv2.IMREAD_GRAYSCALE)
        f2q = cv2.imread(f2q_element,cv2.IMREAD_GRAYSCALE)
        fq = fq.reshape(fq.shape[0]*fq.shape[1],)
        f2q = f2q.reshape(f2q.shape[0]*f2q.shape[1],)
        comparagram = calc_histgram(f2q,fq,"out_"+fq_element.split('.')[0]+ "_"+ f2q_element.split('.')[0]+".txt")
        comparagram[comparagram>255] = 255
        img = Image.fromarray(np.flip(comparagram.astype(np.uint8),0),mode='L')
        #np.savetxt("mono_after.txt",comparagram,fmt="%d")
        img.save("out_"+fq_element.split('.')[0]+ "_"+ f2q_element.split('.')[0]+".jpg")
