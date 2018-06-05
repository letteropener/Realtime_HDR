from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


# place 2 pictures in the dir
# 1st one with high exposure, 2nd one with low exposure
# differ by k (k = 2 in our case)
def _get_images(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir,filename)
        photo_filenames.append(path)
    print(photo_filenames)
    return photo_filenames

def RGB_Split(_photo_filenames):
    RGB_Matrix = []

    for img in _photo_filenames:
        image_data = Image.open(img)
        image_data = np.array(image_data)
        R_Matrix = np.zeros((image_data.shape[0],image_data.shape[1]))
        G_Matrix = np.zeros((image_data.shape[0],image_data.shape[1]))
        B_Matrix = np.zeros((image_data.shape[0],image_data.shape[1]))
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                R_Matrix[i][j] = image_data[i][j][0]
                G_Matrix[i][j] = image_data[i][j][1]
                B_Matrix[i][j] = image_data[i][j][2]
        RGB_Matrix.append([R_Matrix,G_Matrix,B_Matrix])
    return RGB_Matrix

def calc_histgram(_RGB_Matrix):
    R_hist = np.zeros((256,256))
    G_hist = np.zeros((256,256))
    B_hist = np.zeros((256,256))
    for i in range(_RGB_Matrix[0][0].shape[0]):
        for j in range(_RGB_Matrix[0][0].shape[1]):
            if R_hist[int(_RGB_Matrix[0][0][i][j])][int(_RGB_Matrix[1][0][i][j])] < 255:
                R_hist[int(_RGB_Matrix[0][0][i][j])][int(_RGB_Matrix[1][0][i][j])] += 1
    for i in range(_RGB_Matrix[0][1].shape[0]):
        for j in range(_RGB_Matrix[0][1].shape[1]):
            if G_hist[int(_RGB_Matrix[0][1][i][j])][int(_RGB_Matrix[1][1][i][j])] < 255:
                G_hist[int(_RGB_Matrix[0][1][i][j])][int(_RGB_Matrix[1][1][i][j])] += 1
    for i in range(_RGB_Matrix[0][2].shape[0]):
        for j in range(_RGB_Matrix[0][2].shape[1]):
            if B_hist[int(_RGB_Matrix[0][2][i][j])][int(_RGB_Matrix[1][2][i][j])] < 255:
                B_hist[int(_RGB_Matrix[0][2][i][j])][int(_RGB_Matrix[1][2][i][j])] += 1

    return [R_hist,G_hist,B_hist]



def form_final_img(_hist_RGB):
    final_img_array = np.zeros((256, 256, 3))
    for i in range(256):
        for j in range(256):
                final_img_array[i][j][0] = _hist_RGB[0][i][j]
                final_img_array[i][j][1] = _hist_RGB[1][i][j]
                final_img_array[i][j][2] = _hist_RGB[2][i][j]
    return final_img_array

def form_XY_scatterplot(_hist_RGB):
    X = []
    Y = []
    for i in range(256):
        for j in range(256):
            if _hist_RGB[0][i][j] > 0 or _hist_RGB[1][i][j] > 0 \
                or _hist_RGB[2][i][j] > 0:
                X.append(float(i))
                Y.append(float(j))
    return X,Y

Raw_Matrix = RGB_Split(_get_images('img/'))
hist_RGB = calc_histgram(Raw_Matrix)
img_array = form_final_img(hist_RGB)
img = Image.fromarray(np.uint8(np.flip(img_array,0)),mode='RGB')

img_R = Image.fromarray(np.uint8(np.flip(img_array[:,:,0],0),mode='L'))
img_G = Image.fromarray(np.uint8(np.flip(img_array[:,:,1],0),mode='L'))
img_B = Image.fromarray(np.uint8(np.flip(img_array[:,:,2],0),mode='L'))
img.save('result/out.jpg')
img_R.save('result/out_R.jpg')
img_G.save('result/out_G.jpg')
img_B.save('result/out_B.jpg')
