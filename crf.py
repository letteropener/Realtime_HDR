import numpy as np
import sys
from scipy.optimize import curve_fit
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import scipy.ndimage
epsilon=0.00001
class ACS_CRF(object):
    def __init__(self,k):
        self.k=k
        self.a=10
        self.c=0.1
        self.s=1
        #f inverse lookup table
        self.LUT_R = {}
        self.LUT_B = {}
        self.LUT_G = {}

    def __repr__(self):
        return "ACS model with a={}, c={}, s={}".format(self.a,self.c,self.s)
    def acs_func(self,q):
        ret=self.s*(q**self.a/(1+q**self.a))**self.c
        #print(np.where(ret>1))
        #print(ret[np.where(ret>1)])
        try:
            ret[np.where(ret>1)]=1
        except:
            pass
        return ret
    def acs_inverse(self,f_data):
        if f_data>self.s:
            f_data=self.s-epsilon
        if f_data>1:
            f_data=1
        q_a=(f_data**(1/self.c)/(self.s**(1/self.c)-f_data**(1/self.c)))
        return q_a**(1/self.a)
    def acs_compara(self,f_data,a,c,s):
        #for curve fit only
        return (s*f_data*self.k**(a*c))/(f_data**(1/c)*(self.k**a-1)+s**(1/c))**c

    def trim_data(self,f_data,g_data):
        norm_f=f_data[g_data<255]/255.
        norm_g=g_data[g_data<255]/255.
        return norm_f,norm_g
    
    def fit(self,f_data,g_data,_maxfev=2000):
        popt, _ = curve_fit(self.acs_compara, f_data,g_data,maxfev=_maxfev)
        self.a,self.c,self.s=popt
    def generate_LUT(self):
        for f in range(256):
            self.LUT_R[f] = self.acs_inverse(f)
            self.LUT_G[f] = self.acs_inverse(f)
            self.LUT_B[f] = self.acs_inverse(f)


def sigmaF1(comparasum):
    sigma = np.zeros((256,))
    sum_col = np.zeros((256,))
    cdf_col = np.zeros((256,256))
    sum_col_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    result_Q1 = np.zeros((256,))
    result_Q3 = np.zeros((256,))
    IQR = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_col_tmp += comparasum[j][i]
            cdf_col[j][i] = sum_col_tmp
        sum_col_tmp = 0
    np.savetxt('cdf_col.txt',cdf_col,fmt="%d")
    for i in range(256):
        sum_col[i] = np.sum(comparasum[:,i])
        Q1[i] = (sum_col[i])*0.25
        Q3[i] = (sum_col[i])*0.75

        for j in range(256):
            if (Q1[i] == cdf_col[j][i]):
                result_Q1[i] = j
                break
            if (Q1[i] > cdf_col[j][i] and Q1[i] < cdf_col[j+1][i] ):
                result_Q1[i] = j + (Q1[i]-cdf_col[j][i])/(cdf_col[j+1][i] - cdf_col[j][i])
                break
        for j in range(256):
            if (Q3[i] == cdf_col[j][i]):
                result_Q3[i] = j
                break
            if (Q3[i] > cdf_col[j][i] and Q3[i] < cdf_col[j+1][i]):
                result_Q3[i] = j + (Q3[i]-cdf_col[j][i])/(cdf_col[j+1][i] - cdf_col[j][i])
                break

    IQR = result_Q3 - result_Q1
    #print(IQR)
    sigma = np.divide(IQR,1.349)
    return sigma

def sigmaF2(comparasum):
    # sigma of all rows
    sigma = np.zeros((256,))
    sum_row = np.zeros((256,))
    cdf_row = np.zeros((256, 256))
    sum_row_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    result_Q1 = np.zeros((256,))
    result_Q3 = np.zeros((256,))
    IQR = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_row_tmp += comparasum[i][j]
            cdf_row[i][j] = sum_row_tmp
        sum_row_tmp = 0
    np.savetxt('cdf_row.txt', cdf_row, fmt="%d")
    for i in range(256):
        sum_row[i] = np.sum(comparasum[i,:])
        Q1[i] = (sum_row[i])*0.25
        Q3[i] = (sum_row[i])*0.75

        for j in range(256):
            if (Q1[i] == cdf_row[j][i]):
                result_Q1[i] = j
                break
            if (Q1[i] > cdf_row[i][j] and Q1[i] < cdf_row[i][j+1]):
                result_Q1[i] = j + (Q1[i]-cdf_row[i][j])/(cdf_row[i][j+1] - cdf_row[i][j])
                break

        for j in range(256):
            if (Q3[i] == cdf_row[j][i]):
                result_Q3[i] = j
                break
            if (Q3[i] > cdf_row[i][j] and Q3[i] < cdf_row[i][j+1]):
                result_Q3[i] = j + (Q3[i]-cdf_row[i][j])/(cdf_row[i][j+1] - cdf_row[i][j])
                break
    #print(result_Q3)
    #print(result_Q1)
    IQR = result_Q3 - result_Q1
    #print(IQR)
    sigma = np.divide(IQR,1.349)
    return sigma


def smoothsigmas(sigmaf1,sigmaf2,prefix):
    #print(sigmaf1)
    #print(sigmaf2)
    step = 15
    plt.plot(sigmaf1,'r--',label='sigma(col_based)-- unsmoothed',linewidth=1, markersize=1) #color='#FFC0CB'
    smooth_f1 = scipy.ndimage.gaussian_filter(sigmaf1, step)
    plt.plot(smooth_f1,'r',label='sigma(col_based)-- smoothed',linewidth=1, markersize=12)

    plt.plot(sigmaf2,'b--',label='sigma(row_based)-- unsmoothed',linewidth=1, markersize=1) #color='#ADD8E6'
    smooth_f2 = scipy.ndimage.gaussian_filter(sigmaf2, step)
    plt.plot(smooth_f2,'b',label='sigma(row_based)-- smoothed',linewidth=1, markersize=12)
    plt.legend()
    plt.ylim((0,10))
    plt.xlim((0,255))
    plt.savefig('sigmas_{}.svg'.format(prefix), format='svg', dpi=1200)
    #plt.show()
    return smooth_f1,smooth_f2

#def create_f_inverse_LUT(model):
#    for f in range(256):
#       f_inverse_LUT_R[f] = acs_inverse((f + 0.5) / 256., a=_a, c=_c)

def create_CCRF_LUT(model,sigmaf1,sigmaf2,color):
    print("using k={}".format(model.k))
    CCRF = np.zeros((256,256))
    for i in range(256):
        if i%20==0:
            print("Row Completed:({}/256)".format(i))
        for j in range(256):
            CCRF[i][j] = CCRF_argmin(i, j, sigmaf1, sigmaf2,model)
            
    CCRF = round_matrix(CCRF).astype(np.uint8)
    np.savetxt('CCRF_'+color+'_{}.txt'.format(model.k), CCRF, fmt="%d")
    output = Image.fromarray(CCRF, mode='L')
    output.save('CCRF_'+color+'_{}.jpg'.format(model.k))
    return CCRF

def round_matrix(m):
    ret = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            try:
                ret[i][j] = int(round(m[i][j]))
            #assign 0.0/0.0 = 0
            except:
                
                ret[i][j] = 0 
    return ret

def bin_count(comparagram,color="RGB"):
    fit_curve = np.zeros(256)
    for i in range(256):
        indices = np.argmax(comparagram[:,i])
        #print(indices)
        fit_curve[i] = indices
    np.savetxt("fit_" + color + "fit.txt", fit_curve, fmt="%d")
    x_axis = np.array(range(256))

    return fit_curve, x_axis
def unroll(comparagram,color):
    x_data = np.zeros(int(np.sum(comparagram)))
    y_data = np.zeros(int(np.sum(comparagram)))
    data_pos_flag = 0
    for i in range(256):
        for j in range(256):
            if comparagram[i][j] != 0:
                for num in range(int(comparagram[i][j])):
                    y_data[data_pos_flag] = i
                    x_data[data_pos_flag] = j
                    data_pos_flag += 1
    print("total :",np.sum(int(np.sum(comparagram))))
    print("went through :",data_pos_flag)
    #np.savetxt("result/fit_" + color + "fit.txt", fit_curve, fmt="%d")
    #x_axis = np.array(range(256))
    #plt.plot(x_axis[fit_curve > 0],fit_curve[fit_curve > 0], c='black', linewidth=2.0)
    #plt.savefig("result/fit_"+ color + ".jpg")
    #plt.clf()
    return y_data, x_data

def CCRF_argmin(f1,f2,sigmaf1,sigmaf2,model):
    results = np.zeros((256,))
    for i in range(256):
        q = model.LUT_G[i]
        q_2 = q*model.k
        results[i] = ((((f1+0.5)/256. - model.acs_func(q))**model.k)/((sigmaf1[f1])**model.k)) + \
        ((((f2+0.5)/256. - model.acs_func(q_2)) ** model.k) / ((sigmaf2[f2]) ** model.k))
    result = model.LUT_G[np.argmin(results)]
    return model.acs_func(result) * 256. -0.5

if __name__=="__main__":
    import matplotlib.pyplot as plt
    color = 'g'
    comparasum=np.loadtxt("comparasum_R_raw_8.txt")
    comparasum=np.flip(comparasum,0)
    print(comparasum.shape)

    model=ACS_CRF(k=8)
    g_data,f_data=bin_count(comparasum,color)
    assert g_data.shape==f_data.shape
    
    f_data,g_data=model.trim_data(f_data,g_data)
    plt.plot(f_data,g_data)
    plt.show()
    model.fit(f_data,g_data)
    print(model)
    q=np.linspace(0,100,500)
    f=model.acs_func(q)
    plt.plot(q,f)
    #plt.show()

    sigmaf1 = sigmaF1(np.flip(comparasum,0))
    sigmaf2 = sigmaF2(np.flip(comparasum,0))
    
    smoothf1, smoothf2 = smoothsigmas(sigmaf1,sigmaf2,color)
    print("smooth sigma 1",smoothf1)
    model.generate_LUT()

    #MJ
    #LUT_G
    CCRF_LUT = create_CCRF_LUT(model, smoothf1,smoothf2,color)