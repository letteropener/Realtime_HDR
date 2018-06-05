from PIL import Image
import numpy as np
import cv2
import os


from config import *
base = 4
set_id=None
try:
    with open(os.path.join(data_dir,marker_filename)) as fin:
        set_id=fin.read()

except:
    print("Unable to identify the latest img set. Abort")
    raise




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

    np.savetxt(filename + "_B.txt",np.flip(comparagram_B,0),fmt="%d")
    np.savetxt(filename + "_G.txt", np.flip(comparagram_G, 0), fmt="%d")
    np.savetxt(filename + "_R.txt", np.flip(comparagram_R, 0), fmt="%d")
    return comparagram_B,comparagram_G,comparagram_R

img_set_full_dir=os.path.join(data_dir,set_id)
print("Debug: working on {}".format(img_set_full_dir))
photo = os.listdir(photoBaseDir)
k = 2
imgFiles = ["{}.jpg".format(base*k**i) for i in range(20)]
for index in range(1,5):

    
    #imageFiles = os.listdir(os.path.join(photoBaseDir, set_id))
    #print("current folder:{}/{},index: {}/{}".format(count,len(photo),index,4))
    img = ["{}/{}/{}".format(photoBaseDir,set_id,fname) for fname in imgFiles]
    #print(img)
    for i in range(len(img)-index):
        fq_element = img[i]
        f2q_element = img[i+index]
        
        fq = cv2.imread(fq_element)
        f2q = cv2.imread(f2q_element)
        fq_fname=imgFiles[i]
        f2q_fname=imgFiles[i+index]
        B,G,R = calc_histgram(f2q,fq,photoBaseDir + "/"+set_id+"/out_"+fq_fname.split('.')[0]+ "_"+ f2q_fname.split('.')[0])
        B[B>255] = 255
        G[G>255] = 255
        R[R>255] = 255
        img_B = Image.fromarray(np.flip(B.astype(np.uint8),0),mode='L')
        img_G = Image.fromarray(np.flip(G.astype(np.uint8), 0), mode='L')
        img_R = Image.fromarray(np.flip(R.astype(np.uint8), 0), mode='L')
        #np.savetxt("mono_after.txt",comparagram,fmt="%d")
        img_B.save(photoBaseDir + "/"+ set_id+"/out_"+fq_fname.split('.')[0]+ "_"+ f2q_fname.split('.')[0]+"_B.jpg")
        img_G.save(photoBaseDir + "/" +set_id+"/out_" + fq_fname.split('.')[0] + "_" + f2q_fname.split('.')[0] + "_G.jpg")
        img_R.save(photoBaseDir + "/" +set_id+"/out_" + fq_fname.split('.')[0] + "_" + f2q_fname.split('.')[0] + "_R.jpg")

print("You may run comparasum now")