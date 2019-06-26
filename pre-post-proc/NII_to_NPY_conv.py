import nibabel as nib
import numpy as np
import os

# file_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NII_Train/Hi/"
# save_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY_Train/Hi/"
file_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NII_Test/Lo/"
save_path = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/lo/"

file_list = os.listdir(file_path)

for img in file_list:
    vol = np.float32(nib.load(file_path + img).get_fdata())
    np.save(save_path + img[:-4], vol)

    print(img[:-4])
