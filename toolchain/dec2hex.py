import numpy as np


if __name__ == "__main__":
    CCRF_LUT = np.loadtxt('LUTs/CCRF_mono_2x.txt')
    #CCRF_LUT_HEX = np.zeros((CCRF_LUT[:][1].size,CCRF_LUT[1][:].size),dtype='str')
    CCRF_LUT_HEX = np.array(range(CCRF_LUT.size), dtype=str).reshape(CCRF_LUT[:][1].size,CCRF_LUT[1][:].size)
    for i in range(0,CCRF_LUT[1][:].size):
        for j in range(0,CCRF_LUT[:][1].size):
            #print(hex(int(CCRF_LUT[i][j])))
            CCRF_LUT_HEX[i][j] = hex(int(CCRF_LUT[i][j]))
    #print(CCRF_LUT_HEX[255][111])
    print(CCRF_LUT_HEX[255][255])
    print(CCRF_LUT[255][255])
    #np.savetxt('CCRF_mono_2x_hex.txt', CCRF_LUT_HEX, fmt="%s")