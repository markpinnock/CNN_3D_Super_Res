import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sci
import sys

sys.path.append('..')

from utils.functions import imgLoader
from utils.DataAugKeras import DataAugmentationKeras
from utils.DataAugNumpy import DataAugmentationNumpy
from utils.DataAugScipy import DataAugmentationScipy

FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

os.chdir(FILE_PATH)
hi_list = os.listdir('Hi/')
lo_list = os.listdir('Lo/')
hi_list.sort()
lo_list.sort()

hi_vol, lo_vol = imgLoader(hi_list, lo_list, [0, 1, 2, 3])

# DataAug = DataAugmentationKeras([4, 128, 128, 12, 1], 0.3)
# DataAug = DataAugmentationNumpy([4, 128, 128, 12, 1], 1)
DataAug = DataAugmentationScipy([4, 128, 128, 12, 1])

new_hi_vol, new_lo_vol = DataAug.warpImg(
    hi_vol, lo_vol, flip=None, rot=24, scale=0.25, shear=None)

# new_hi_vol, new_lo_vol = DataAug.transform(hi_vol, lo_vol)

for i in range(1):
    fig, axs = plt.subplots(2, 4)
    # axs[0, 0].imshow(hi_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 0].imshow(new_hi_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 1].imshow(new_hi_vol[1, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 2].imshow(new_hi_vol[2, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 3].imshow(new_hi_vol[3, :, :, i, 0].T, origin='lower', cmap='gray')
    # axs[1, 0].imshow(lo_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 0].imshow(new_lo_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 1].imshow(new_lo_vol[1, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 2].imshow(new_lo_vol[2, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 3].imshow(new_lo_vol[3, :, :, i, 0].T, origin='lower', cmap='gray')
    plt.show()