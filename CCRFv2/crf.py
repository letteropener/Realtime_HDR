import os
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
from util import comparasum_row_iqr
from util import SIGMA_THRESHOLD
import matplotlib.pyplot as plt
from PIL import Image
class ACModel(object):
    def __init__(self,a=1,c=1,k=2,channel='mono',bits=8):
        size=2**bits
        self.a=a
        self.c=c
        self.k=k
        self.comparasum=None
        self.f_inverse_LUT=np.zeros(size)
        self.channel=channel
        self.sigma1=None#by row
        self.sigma2=None#by col
        self.smooth_sigma1=None
        self.smooth_sigma2=None #gaussian smoothed sigmas
        self.kq=None
        self.LUT=np.zeros((size,size),dtype=np.uint8)

    def __repr__(self):
        return "AC camera model with a={}, c={} for channel: {}".format(self.a,self.c,self.channel)
    def load_comparasum(self,file_name):
        self.comparasum=np.loadtxt(file_name,dtype=np.int32)
        self.comparasum=np.flip(self.comparasum,axis=0)

#Region math model functions
    def f(self,q):
        '''
        calculate f(q)=(q^a/(1+q^a))^c, where f in [0,1]
        '''
        
        q_a=q**self.a
        return (q_a/(1+q_a))**self.c

    def f_inverse(self,f_value):
        
        f_croot=f_value**(1.0/self.c)
        return (f_croot/(1+f_croot))**(1/self.a)

    def f_compara(self,f_value):
        '''same as f_compara_fit, but uses self.a and self.c'''
        return self.f_compara_fit(f_value,self.a,self.c)
    def f_compara_fit(self,f_value,a,c):
        '''calculate g(f)=f(k*q), with given a and c
           f,g in [0,1]
            For curve fit only
        '''
        k_a=self.k**a
        k_ac=self.k**(a*c)

        f_croot=np.power(f_value,1.0/c)

        return (f_value*k_ac)/(f_croot*(k_a-1)+1)**c
    def create_f_inverse_LUT(self):
        f_values=np.arange(256)/255.0
        self.f_inverse_LUT=self.f_inverse(f_values)
    def fit(self,f_data,g_data):
        popt,_ = curve_fit(self.f_compara_fit,f_data,g_data)
        self.a,self.c=popt
        print("Done fitting")
        print(self.__repr__())

    def calc_kq(self,verbose=False):
        self.kq=np.zeros(self.f_inverse_LUT.shape)

        for f in range(len(self.kq)):
            q=self.f_inverse_LUT[f]
            if verbose:
                print("q={}".format(q))
            self.kq[f]=int(self.f(self.k*q)*255)
        if verbose:
            print("---VERBOSE:kq lookup table---")
            print(self.kq)
            print("---END OF VERBOSE OUTPUT---")
#end of Region math model
# 
       
    def unroll_comparasum(self):
        assert self.comparasum is not None
        x_data = np.zeros(int(np.sum(self.comparasum)))
        y_data = np.zeros(int(np.sum(self.comparasum)))
        data_pos_flag = 0
        for i in range(256):
            for j in range(256):
                if self.comparasum[i][j] != 0:
                    for _ in range(int(self.comparasum[i][j])):
                        y_data[data_pos_flag] = i
                        x_data[data_pos_flag] = j
                        data_pos_flag += 1
        print("total :",np.sum(int(np.sum(self.comparasum))))
        print("went through :",data_pos_flag)
        
        return {'g':y_data/255.0, 'f':x_data/255.0}



#Region LUT
#TODO: Check and implement these functions
    def calc_sigma(self):
        #sigma = iqr/1.349
        self.sigma1=comparasum_row_iqr(self.comparasum.T)/1.349
        self.sigma2=comparasum_row_iqr(self.comparasum)/1.349

    def calc_smooth_sigmas(self,kernel_size=15):
        if self.sigma1 is None or self.sigma2 is None:
            raise ValueError("Sigmas are not calculated")
        self.smooth_sigma1=gaussian_filter1d(self.sigma1,kernel_size)
        self.smooth_sigma2=gaussian_filter1d(self.sigma2,kernel_size)

    def LUT_cell(self,i,j,verbose=False):
        ''' calculate LUT entry at location i-th row, j-th col
            uses sigmas within the instance
         '''
        

        if i<0 or i>=self.comparasum.shape[0] or j<0 or j>=self.comparasum.shape[1]:
            raise ValueError("Invalid LUT entry coordinates ({},{})".format(i,j))
        if verbose:
            print("Sigma1 = {}, Sigma2 = {}".format(self.smooth_sigma1[i],self.smooth_sigma2[j]))

        if self.smooth_sigma1[i]<SIGMA_THRESHOLD:
            return j
        if self.smooth_sigma2[j]<SIGMA_THRESHOLD:
            return i
        candidates=np.zeros(self.sigma1.shape)
        for f1 in range(len(candidates)):
            #given f(q)=f1, find coresponding f(kq)=f2
            f2=self.kq[f1]
            candidates[f1]=(i-f1)**2/self.smooth_sigma1[i]**2+(j-f2)**2/self.smooth_sigma2[j]**2
        if verbose:
            print("LUT cell candidates are:", candidates)
        return np.argmin(candidates)

        

    def calc_LUT(self):
        if self.smooth_sigma1 is None or self.smooth_sigma2 is None:
            raise ValueError("Sigmas are not calculated")
        n=self.comparasum.shape[0]
        for i in range(n):
            for j in range(n):
                self.LUT[i][j]=self.LUT_cell(i,j)
#END of Region LUT

#Region utils
    def plot_sigmas(self):
        plt.clf()
        plt.ylim([0,10])
        plt.plot(self.sigma1,'b',label="raw")
        plt.plot(self.smooth_sigma1,'r',label="smoothed")
        plt.legend()
        plt.savefig("sigma1_{}_{}.jpg".format(self.channel,self.k))
        plt.clf()
        plt.ylim([0,10])
        plt.plot(self.sigma2,'b',label="raw")
        plt.plot(self.smooth_sigma2,'r',label="smoothed")
        plt.legend()
        plt.savefig("sigma2_{}_{}.jpg".format(self.channel,self.k))
        plt.clf()
    def save_sigmas_txt(self):
        np.savetxt("sigma1_raw_{}_{}.txt".format(self.channel,self.k),self.sigma1,fmt="%.6f")
        np.savetxt("sigma2_raw_{}_{}.txt".format(self.channel,self.k),self.sigma2,fmt="%.6f")
        np.savetxt("sigma1_smooth_{}_{}.txt".format(self.channel,self.k),self.smooth_sigma1,fmt="%.6f")
        np.savetxt("sigma2_smooth_{}_{}.txt".format(self.channel,self.k),self.smooth_sigma2,fmt="%.6f")
    def plot_LUT(self,outdir=None):
        if outdir is not None:
            fname=os.path.join(outdir,"LUT_{}_{}.jpg".format(self.k,self.channel))
        else:
            fname="LUT_{}_{}.jpg".format(self.k,self.channel)
        img=Image.fromarray(self.LUT,mode='L')
        img.save(fname)
    def save_LUT_txt(self,outdir=None):
        if outdir is not None:
            fname=os.path.join(outdir,"LUT_{}_{}.txt".format(self.k,self.channel))
        else:
            fname="LUT_{}_{}.txt".format(self.k,self.channel)
        np.savetxt(fname,self.LUT,delimiter=' ',fmt="%d")
    
    def load_LUT(self,fname):
        self.LUT=np.loadtxt(fname,dtype=np.uint8)

    
#end of Region utils

if __name__=="__main__":
    model=ACModel(5,0.5,k=8,channel='G')
    model.load_comparasum("../comparasum_G_raw_8.txt")
    data=model.unroll_comparasum()
    f_data=data['f']
    g_data=data['g']
    model.fit(f_data,g_data)
    
    model.create_f_inverse_LUT()
    model.calc_kq()
    model.calc_sigma()
    model.calc_smooth_sigmas()
    model.save_sigmas_txt()
    model.plot_sigmas()
    

    #print(model.LUT_cell(50,200,True))
    model.calc_LUT()
    model.plot_LUT()
    model.save_LUT_txt()
    