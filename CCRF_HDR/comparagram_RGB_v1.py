from PIL import Image
import numpy as np
import cv2


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


#photos = ["15.625.jpg","31.25.jpg","62.5.jpg","125.jpg","250.jpg","500.jpg","1000.jpg","2000.jpg",
#          "4000.jpg","8000.jpg","16000.jpg","32000.jpg","64000.jpg","128000.jpg"
#          ]
#photos = ["15.625.jpg","125.jpg","1000.jpg","8000.jpg","64000.jpg"]
base = 4
k = 2
exposures = [base*k**i for i in range(18)]
photos = ['{}.jpg'.format(expo) for expo in exposures]
for i in range(len(photos)-1):
        fq_element = photos[i]
        f2q_element = photos[i+1]
        fq = cv2.imread(fq_element)
        f2q = cv2.imread(f2q_element)
        B,G,R = calc_histgram(f2q,fq,"out_"+fq_element.split('.')[0]+ "_"+ f2q_element.split('.')[0])
        B[B>255] = 255
        G[G>255] = 255
        R[R>255] = 255
        img_B = Image.fromarray(np.flip(B.astype(np.uint8),0),mode='L')
        img_G = Image.fromarray(np.flip(G.astype(np.uint8), 0), mode='L')
        img_R = Image.fromarray(np.flip(R.astype(np.uint8), 0), mode='L')
        #np.savetxt("mono_after.txt",comparagram,fmt="%d")
        img_B.save("out_"+fq_element.split('.')[0]+ "_"+ f2q_element.split('.')[0]+"_B.jpg")
        img_G.save("out_" + fq_element.split('.')[0] + "_" + f2q_element.split('.')[0] + "_G.jpg")
        img_R.save("out_" + fq_element.split('.')[0] + "_" + f2q_element.split('.')[0] + "_R.jpg")
