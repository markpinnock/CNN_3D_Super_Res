import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import sys

sys.path.append('..')

from utils.Gemima_Utils import diceIndexCalc


# FILE_PATH = "F:/PhD/Super_Res_Data/Toshiba_Vols/NII_Test/UCLH_11700946/"
FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"
file_list = os.listdir(FILE_PATH)
seg_list = [vol for vol in file_list if 'seg' in vol]
seg_list.sort()

seg0 = nrrd.read(FILE_PATH + seg_list[0])
seg1 = nrrd.read(FILE_PATH + seg_list[1])
seg2 = nrrd.read(FILE_PATH + seg_list[2])
seg0 = seg0[0]
seg1 = seg1[0]
seg2 = seg2[0]
seg0 = seg0[0:2, :, :]
seg1 = seg1[0:2, :, :]
seg2 = seg2[0:2, :, :]
dice_int = diceIndexCalc(seg0, seg1)
dice_out = diceIndexCalc(seg0, seg2)

print(diceIndexCalc(seg0, seg1))
print(diceIndexCalc(seg0, seg2))
# print(diceIndexCalc(seg1[1, ...], seg2[1, ...]))

# for i in range(0, 2):
#     for j in range(0, seg.shape[3]):
#         if np.sum(seg[i, :, :, j]) == 0:
#             continue
        
#         if j % 10 != 0:
#             continue

#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(np.fliplr((vol[:, :, j] * seg[i, :, :, j]).T), cmap='gray', origin='lower')
#         ax[1].imshow(np.fliplr(vol[:, :, j].T), cmap='gray', origin='lower')
#         plt.show()
