import sys
import numpy as np
EPSILON=sys.float_info.epsilon
DELTA=0.00001
PIXEL_DEPTH=256.0
PIXEL_MAX=PIXEL_DEPTH-1
class LUTFactory(object):
    def __init__(self,a,c):
        self.a=a
        self.c=c
        self.f_inverse_LUT=None
        self.f_derivative_LUT={}
    def setACValue(self,a,c):
        self.a=a
        self.c=c

    def f(self,q):
        #the camera response function(CRF) f(q)=(q^a/(1+q^))^c
        #returns image space pixel value (0-PIXEL_MAX)

        q_a=np.power(q,self.a)#q^a
        f_normalized=np.power((q_a/(1+q_a)),self.c)
        return np.rint(f_normalized*PIXEL_MAX)

    def f_inverse(self,pixel):
        #the inverse of camera response function
        #@param pixel: pixel value between 0-255 in image space
        #@return lightspace value q
        pixel_normalized=np.asarray(pixel)/PIXEL_MAX
        
        pixel_cthroot=np.power(pixel_normalized,1/self.c)
        pixel_cthroot[pixel_cthroot>=1]=1-EPSILON
        q_a=pixel_cthroot/(1-pixel_cthroot)
        return np.power(q_a,1.0/self.a)

    def calc_f_inverse_LUT(self):
        #calculate a lookup table for given a,c values
        pixels=np.arange(PIXEL_DEPTH)
        self.f_inverse_LUT=self.f_inverse(pixels)

    def f_confidence(self,pixel,ki):
        #calculate the confidence, i.e. the derivative of the CRF at a given pixel value against ln(q), with a certain ki
        # @param pixel: image space pixel values
        # @param ki: the exposure coefficent associated with said pixel value. Modifies the confidence output. See page 117 of `Intelligent Image Processing`
        #df/dq=c*((a*q^(a - 1))/(q^a + 1) - (a*q^a*q^(a - 1))/(q^a + 1)^2)*(q^a/(q^a + 1))^(c - 1)
        assert ki>0
        print(ki)
        pixel=np.asarray(pixel)
        if self.f_inverse_LUT is None:
            self.calc_f_inverse_LUT()
        q=self.f_inverse_LUT[pixel]*ki
        '''def quick_f(q_value):
            q_a=q_value**self.a
            return np.power(q_a/(1+q_a),self.c)
        f_deriv=(quick_f(q+DELTA)-quick_f(q))/DELTA
        return f_deriv*ki'''
        f_deriv=self.c*((self.a*q**(self.a - 1))/(q**self.a + 1) - (self.a*q**self.a*q**(self.a - 1))/(q**self.a + 1)**2) * (q**self.a/(q**self.a + 1))**(self.c - 1)
        f_deriv[0]=0
        return f_deriv*ki
    def create_HDR_LUT(self,conf1,conf2):
        size=int(PIXEL_DEPTH)
        ret=np.zeros((size,size))
        for p1 in range(size):
            q1=self.f_inverse_LUT[p1]
            for p2 in range(size):
                q2=self.f_inverse_LUT[p2]
                weights=[conf1[p1],conf2[p2]]
                if sum(weights)==0:
                    weights=[1,1]
                qhat=np.average([q1,q2],weights=weights)
                ret[p1][p2]=self.f(qhat)            
        return ret
    def create_HDR_LUT_star(self,arg):
        return self.create_HDR_LUT(*arg)
                 
def hdr_merge(LUT,img1,img2):
    assert img1.shape==img2.shape
    shape=img1.shape
    flat_img1=img1.flatten()
    flat_img2=img2.flatten()
    flat_ret=[]
    for i,j in zip(flat_img1,flat_img2):
        flat_ret.append(LUT[i][j])
    flat_ret=np.asarray(flat_ret)
    ret=np.reshape(flat_ret,shape)
    return ret




if __name__=='__main__':
    import matplotlib.pyplot as plt
    import cv2
    fac_b=LUTFactory(7.715307360250536,0.1338181325596138)
    fac_g=LUTFactory(12.041222244933214,0.08311990140501978)
    fac_r=LUTFactory(10.513381309047697,0.09578753831251556)
    
    raw_pix=np.arange(256)

    conf1_b=fac_b.f_confidence(raw_pix,1)
    conf2_b=fac_b.f_confidence(raw_pix,4)
    conf1_g=fac_g.f_confidence(raw_pix,1)
    conf2_g=fac_g.f_confidence(raw_pix,4)
    conf1_r=fac_r.f_confidence(raw_pix,1)
    conf2_r=fac_r.f_confidence(raw_pix,4)
    
    
    lut_b=fac_b.create_HDR_LUT(conf1_b,conf2_b)    
    np.savetxt("lutb.txt",lut_b,fmt="%d")
    lut_g=fac_g.create_HDR_LUT(conf1_g,conf2_g)    
    np.savetxt("lutg.txt",lut_g,fmt="%d")
    lut_r=fac_r.create_HDR_LUT(conf1_r,conf2_r)    
    np.savetxt("lutr.txt",lut_r,fmt="%d")

    img1=cv2.imread('2048.jpg')
    img2=cv2.imread('8192.jpg')
    img1_b,img1_g,img1_r=cv2.split(img1)
    
    img2_b,img2_g,img2_r=cv2.split(img2)
    print(img1_b.shape)
    out_b=hdr_merge(lut_b,img1_b,img2_b)
    out_g=hdr_merge(lut_g,img1_g,img2_g)
    out_r=hdr_merge(lut_r,img1_r,img2_r)
    out=cv2.merge([out_b,out_g,out_r])
    cv2.imwrite("out.jpg",out)
    #use lut
