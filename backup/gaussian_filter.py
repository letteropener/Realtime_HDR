import numpy as np
from scipy import ndimage
import cv2
from numba import vectorize,float64,int32,guvectorize
from timeit import default_timer as timer
import math
from PIL import Image

PI = 3.1415926

class kernel_class:
    def __init__(self, radius=1, sigma=5):
        #change the parameters
        self.radius = radius
        self.sigma = sigma
        self.top_left_kernel = get_top_left_corner_weight(self.radius,self.sigma)
        self.top_right_kernel = get_top_right_corner_weight(self.radius, self.sigma)
        self.bot_left_kernel = get_bot_left_corner_weight(self.radius, self.sigma)
        self.bot_right_kernel = get_bot_right_corner_weight(self.radius, self.sigma)
        self.left_kernel = get_left_weight(self.radius,self.sigma)
        self.right_kernel = get_right_weight(self.radius,self.sigma)
        self.top_kernel = get_top_weight(self.radius,self.sigma)
        self.bot_kernel = get_bot_weight(self.radius,self.sigma)
        self.mid_kernel = get_weight(self.radius, self.sigma)
    
def get_weight(radius=1,sigma=5):
    sigma2=sigma**2
    n=radius*2+1 #2+1 = 3 radius
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((n,n)) # 3x3 dist
    weight=np.zeros((n,n)) # 3x3 weight
    for i in range(n):
        for j in range(n): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    
    return weight/np.sum(weight) #final weight matrix

def get_top_left_corner_weight(radius=1,sigma=5):
    sigma2 = simga**2
    n = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((n,n)) # 2x2 dist
    weight=np.zeros((n,n)) # 2x2 weight
    for i in range(n):
        for j in range(n): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(n):
        for l in range(n):
            if k == 0 and l == 0:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_top_right_corner_weight(radius=1,sigma=5):
    sigma2 = simga**2
    n = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((n,n)) # 2x2 dist
    weight=np.zeros((n,n)) # 2x2 weight
    for i in range(n):
        for j in range(n): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(n):
        for l in range(n):
            if k == 1 and l == 0:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_bot_left_corner_weight(radius=1,sigma=5):
    sigma2 = simga**2
    n = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((n,n)) # 2x2 dist
    weight=np.zeros((n,n)) # 2x2 weight
    for i in range(n):
        for j in range(n): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(n):
        for l in range(n):
            if k == 0 and l == 1:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_bot_right_corner_weight(radius=1,sigma=5):
    sigma2 = simga**2
    n = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((n,n)) # 2x2 dist
    weight=np.zeros((n,n)) # 2x2 weight
    for i in range(n):
        for j in range(n): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(n):
        for l in range(n):
            if k == 1 and l == 1:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_bot_weight(radius=1,sigma=5):
    sigma2 = simga**2
    r = radius*2
    c = radius*2+1
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((r,c)) # 2x3 dist
    weight=np.zeros((r,c)) # 2x3 weight
    for i in range(r):
        for j in range(c): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(r):
        for l in range(c):
            if k == 1:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_top_weight(radius=1,sigma=5):
    sigma2 = simga**2
    r = radius*2
    c = radius*2+1
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((r,c)) # 2x3 dist
    weight=np.zeros((r,c)) # 2x3 weight
    for i in range(r):
        for j in range(c): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(r):
        for l in range(c):
            if k == 0:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_left_weight(radius=1,sigma=5):
    sigma2 = simga**2
    r = radius*2+1
    c = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((r,c)) # 2x3 dist
    weight=np.zeros((r,c)) # 2x3 weight
    for i in range(r):
        for j in range(c): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(r):
        for l in range(c):
            if l == 0:
                weight[k][l] = weight[k][l]/sum_weight
            else:
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def get_right_weight(radius=1,sigma=5):
    sigma2 = simga**2
    r = radius*2+1
    c = radius*2
    dem=2*np.pi*sigma2 #denominator
    dist=np.zeros((r,c)) # 2x3 dist
    weight=np.zeros((r,c)) # 2x3 weight
    for i in range(r):
        for j in range(c): #iterate through the matrix
            #calculate coordinates relative to center
            x=i-radius
            y=j-radius
            dist[i][j]=(x**2+y**2) #distance 
    
    weight=np.exp(dist/(-2*sigma2))/dem
    sum_weight = np.sum(weight)

    for k in range(r):
        for l in range(c):
            if l == 1:
                weight[k][l] = weight[k][l]/sum_weight
            else
                weight[k][l] = weight[k][l]/sum_weight * 2
    return weight

def gaussian_blur(img, classBlur):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(img.height):
        for j in range(img.width):
            if i == 0 and j == 0: #top left corner
                img[i][j] = kernel_filter(img[i:i+1][j:j+1],classBlur.top_left_kernel)
            elif i == 0 and j == width:#top right corner
                img[i][j] = kernel_filter(img[i:i+1][j-1:j], classBlur.top_right_kernel)
            elif i == height and j == 0: #bottom left corner
                img[i][j] = kernel_filter(img[i-1:i][j:j+1], classBlur.bot_left_kernel)
            elif i == height and j == width:#bottom right corner
                img[i][j] = kernel_filter(img[i-1:i][j-1:j], classBlur.bot_right_kernel)
            elif i == 0 and j != 0: #top row
                img[i][j] = kernel_filter(img[i:i+1][j-1:j+1],classBlur.top_kernel)
            elif i == height and j != 0: #bottom row
                img[i][j] = kernel_filter(img[i-1:i][j-1:j+1],classBlur.bot_kernel)
            elif i != 0 and j == 0: #left column
                img[i][j] = kernel_filter(img[i-1:i+1][j:j+1],classBlur.left_kernel)
            elif i != 0 and j == width: #right column
                img[i][j] = kernel_filter(img[i-1:i+1][j-1:j],classBlur.right_kernel)
            else:#middle area
                img[i][j] = kernel_filter(img[i-1:i+1][j-1:j+1],classBlur.mid_kernel)
    return img






def kernel_filter(img,kernel):
    '''Filter an array of the same size as the kernel'''
    assert img.shape==kernel.shape
    return np.sum(np.multiply(img,kernel))


@vectorize([float64(float64,float64)],target='parallel')
def kernal_mulitiple(subimage,kernal):
        return np.multiply(subimage,kernal)



@guvectorize([(float64[:],float64[:],float64[:])],'(n),(n)->(n)',target='parallel')
def kernal_mulitiple1(subimage,kernal,output):
    for i in range(1):
        output[0] = np.sum(np.multiply(subimage,kernal))
    #print((np.multiply(subimage,kernal)))
    #print(np.sum((np.multiply(subimage,kernal))))

'''
@vectorize([float64(float64,float64)],target='parallel')
def filter_result(input,radius):
    for i in (input.shape[0] - 3 + 1):
        for j in (input.shape[1] - 3 + 1):
            center = kernal_mulitiple(input[i:i+2][j:j+2])
    return center
'''


def Matrixmaker(r):
    summat = 0
    start = timer()
    ma = np.empty((2*r+1,2*r+1))
    for i in range(0,2*r+1):
        for j in range(0,2*r+1):
            gaussp = (1/(2*PI*(r**2))) * math.e**(-((i-r)**2+(j-r)**2)/(2*(r**2)))
            ma[i][j] = gaussp
            summat += gaussp
    #print(ma)
    #print(summat)
    for i in range(0,2*r+1):
        for j in range(0,2*r+1):
            ma[i][j] = ma[i][j]/summat
    end_time = timer() - start
    print("gauss kernel took %f seconds/frame" % end_time)
    return ma

def newrgb(ma,img,r):#生成新的像素rgb矩阵
    start = timer()
    img_width = img.shape[0]
    img_height = img.shape[1]
    newr = np.empty((img_width,img_height))
    newg = np.empty((img_width,img_height))
    newb = np.empty((img_width,img_height))
    nr = img[:, :, 0]
    ng = img[:, :, 0]
    nb = img[:, :, 0]
    for i in range(r+1,img_width-r):
        for j in range(r+1,img_height-r):
            o = 0
            for x in range(i-r,i+r+1):
                p = 0
                for y in range(j-r,j+r+1):
                    #print("x{},y{},o{},p{}".format(x,y,o,p))
                    newr[i][j] += nr[x][y]*ma[o][p]
                    newg[i][j] += ng[x][y]*ma[o][p]
                    newb[i][j] += nb[x][y]*ma[o][p]
                    p += 1
                o += 1
    end_time = timer() - start
    print("calc 3 channel took %f seconds/frame" % end_time)
    return newr,newg,newb


class MyGaussianBlur():
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius = radius
        self.sigema = sigema
        # 高斯的计算公式

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2
        # 得到滤波模版

    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all
        # 滤波函数

    def filter(self, image, template):
        arr = np.array(image)
        height = arr.shape[0]
        width = arr.shape[1]
        newData = np.zeros((height, width))
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                t = arr[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                a = np.multiply(t, template)
                newData[i, j] = a.sum()
        newImage = Image.fromarray(newData)
        return newImage

if __name__=="__main__":
    r = 1  # 模版半径，自己自由调整
    s = 50  # sigema数值，自己自由调整
    GBlur = MyGaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
    temp = GBlur.template()  # 得到滤波模版
    im = Image.open('2.jpg')
    #im = im.convert('L')# 打开图片
    image = GBlur.filter(np.mean(im, axis=2), temp)  # 高斯模糊滤波，得到新的图片
    image.show()  # 图片显示
    '''
    ma = Matrixmaker(1)
    img = cv2.imread('1.jpg')
    print(img.shape)
    nr,ng,nb = newrgb(ma, img, 1)
    print(nr)
    kernel=get_weight(1,5)
    '''
    #print(kernel)
    #data=np.array([13,16,17,26,28,29,37,35,38])
    data = np.array([1., 1, 2, 1])
    data=data.reshape((2,2))
    #print(kernel_filter(data,kernel))
    #print(data)


    start = timer()

    blured_pixels = ndimage.filters.gaussian_filter(np.mean(im, axis=2), 5)
    blured = Image.fromarray(blured_pixels)
    blured.show()
    #print(blured_pixels)
    end_time = timer() - start
    print("ldr_tonemapping took %f seconds/frame" % end_time)

    start = timer()
    res = np.zeros((9,))
    #kernal_mulitiple1(data.reshape((9,)), kernel.reshape((9,)),res)
    end_time = timer() - start
    #print("ldr_tonemapping took %f seconds/frame" % end_time)
    #print(res[0])
