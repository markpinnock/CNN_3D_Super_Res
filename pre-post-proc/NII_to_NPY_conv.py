import nibabel as nib
import numpy as np
import os


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NII_Test/Hi/"
SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/Hi/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/real_test_imgs/Lo/"
# SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/lo/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/real_test_imgs/Lo/"
    SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/lo/"

file_list = os.listdir(FILE_PATH)

for img in file_list[-30:]:
    vol = np.float32(nib.load(FILE_PATH + img).get_fdata())
    np.save(SAVE_PATH + img[:-4], vol)

    print(img[:-4])
