import scipy
from scipy import ndimage
import numpy as np
import cv2
from numba import vectorize,float64
from timeit import default_timer as timer

@vectorize([float64(float64,float64,float64)],target='parallel')
def parallel_pixels(unscaled_output,lower_limit,upper_limit):
    return (unscaled_output - lower_limit) / (upper_limit - lower_limit)

@vectorize([float64(float64,float64,float64)],target='parallel')
def contrast_enhance(pixels, blurred_pixels, intensity):
    x = (pixels - blurred_pixels) * intensity
    return x / (np.abs(x) + 1)

def ldr_tonemap_rgb_image(img, power, radius):
    blured_pixels = ndimage.filters.gaussian_filter(np.mean(img, axis=2), radius)
    intensity = power * power * 0.1

    for channel in range(img.shape[2]):
        pixels = img[:, :, channel]
        #print(pixels.shape)
        unscaled_output = contrast_enhance(pixels, blured_pixels, intensity)

        lower_limit = contrast_enhance(0.0, blured_pixels, intensity)
        upper_limit = contrast_enhance(1.0, blured_pixels, intensity)

        pixels = (unscaled_output - lower_limit) / (upper_limit - lower_limit)

        img[:, :, channel] = pixels
    return img


img = cv2.imread('2.jpg')
skimage_response = ndimage.filters.gaussian_filter(img, 5)
cv2_response = cv2.GaussianBlur(img, (33, 33), 5)
#print(skimage_response)
#print(cv2_response)
#cv2.imwrite('test.jpg',skimage_response)
#cv2.imwrite('test1.jpg',cv2_response)
start = timer()
img = ldr_tonemap_rgb_image(img, 5, 50)
img = ldr_tonemap_rgb_image_cv2(img, 5, 50)
end_time = timer() - start
print("ldr_tonemapping took %f seconds/frame" % end_time)
scipy.misc.imsave("tono_mapped1.jpg", img * 255.0)