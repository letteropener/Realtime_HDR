import os
import numpy as np

LUT_FOLDER="LUTs"

if __name__=="__main__":
    if os.path.isdir(LUT_FOLDER) is False:
        raise ValueError("LUTs folder: {} does not exist".format(LUT_FOLDER))
    
    for item in os.listdir(LUT_FOLDER):
        fullpath=os.path.join(LUT_FOLDER,item)
        if os.path.isfile(fullpath) and item.endswith(".txt"):
            lut=np.loadtxt(fullpath,dtype=np.uint8)
            hex_lut=lut.tobytes()
            newfile=os.path.join(LUT_FOLDER,item)+".hex"
            print(newfile)
            with open(newfile,'wb') as fout:
                fout.write(hex_lut)