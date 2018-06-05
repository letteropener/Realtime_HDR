import numpy as np
from run_CCRF_RGB import parallel_CCRF_B,parallel_CCRF_G,parallel_CCRF_R
from run_CCRF_mono import ldr_tonemap_rgb_image,ldr_tonemap_rgb_image_cv2
import timeit
import cv2

###################
# Note:
# Assumed inputs are 1.jpg,2.jpg,3.jpg,4.jpg in a folder named "inputs"
# Outputs are stored in a folder named "outputs"
# the script does not check if these folders exist

def generate(low_frame,high_frame):
    #Assumes BGR channel arrangement align with opencv
    assert low_frame.shape == high_frame.shape
    ret=np.zeros(low_frame.shape)
    ret[:,:,0]=parallel_CCRF_B(high_frame[:,:,0],low_frame[:,:,0])
    ret[:,:,1]=parallel_CCRF_G(high_frame[:,:,1],low_frame[:,:,1])
    ret[:,:,2]=parallel_CCRF_R(high_frame[:,:,2],low_frame[:,:,2])
    
    return ret

def save_fig(low,high,id,prefix):
    start=timeit.default_timer()
    fig=generate(low,high)
    end=timeit.default_timer()
    print("benchmark: {}".format(end-start))
    fname="outputs/{}_{}.jpg".format(prefix,id)
    cv2.imwrite(fname,fig)
    return fname,fig


    


if __name__=="__main__":
    

    f1=cv2.imread("inputs/1.jpg",cv2.IMREAD_COLOR )
    f2=cv2.imread("inputs/2.jpg",cv2.IMREAD_COLOR )
    f3=cv2.imread("inputs/3.jpg",cv2.IMREAD_COLOR )
    f4=cv2.imread("inputs/4.jpg",cv2.IMREAD_COLOR )
    

    
    print("Input loaded")
    f1_1_name,dummy=save_fig(f1,f2,"1","layer1")
    
    f1_2_name,dummy=save_fig(f2,f3,"2","layer1")
    f1_3_name,dummy=save_fig(f3,f4,"3","layer1")
    
    f1_1=cv2.imread(f1_1_name)
    f1_2=cv2.imread(f1_2_name)
    f1_3=cv2.imread(f1_3_name)
    print("layer1 done and loaded")

    f2_1_name,dummy=save_fig(f1_1,f1_2,'1','layer2')
    f2_2_name,dummy=save_fig(f1_2,f1_3,"2","layer2")
    
    f2_1=cv2.imread(f2_1_name)
    f2_2=cv2.imread(f2_2_name)
    print("layer2 done and loaded")
    f3_1_name,final_frame=save_fig(f2_1,f2_2,"1","layer3")
    #f3_1=cv2.imread(f3_1_name)
    #f3_1_RGB=cv2.cvtColor(f3_1,cv2.COLOR_BGR2RGB)
    print("final layer")
    start=timeit.default_timer()
    mapped=ldr_tonemap_rgb_image(final_frame,0.75,50)
    end=timeit.default_timer()
    print("mapping:{}".format(end-start))
    cv2.imwrite("outputs/mapped.jpg",mapped)
    print("final layer faster")
    start = timeit.default_timer()
    mapped_cv2 = ldr_tonemap_rgb_image_cv2(final_frame, 0.05, 50)
    end = timeit.default_timer()
    print("mapping:{}".format(end - start))
    cv2.imwrite("outputs/mapped_cv2.jpg", mapped_cv2)

