import nrrd
import numpy as np
import os
import scipy.interpolate as sciint


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"
OUT_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/saved_images/007_nc8_ep10_n1026/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"
# SAVE_PATH = "F:/PhD/Super_Res_Data/Toshiba_Vols/NII_Test/"

hi_list = os.listdir(FILE_PATH + 'Hi/')
hi_list.sort()
lo_list = os.listdir(FILE_PATH + 'Lo/')
lo_list.sort()
out_list = os.listdir(OUT_PATH)
out_list.sort()

subject = 'UCLH_11700946'
vol = 9
vol_dims = [512, 512, 12]

hi_1 = np.zeros((512, 512, 12 * vol))
hi_2 = np.zeros((512, 512, 12 * (len(hi_list) - vol)))
int_1 = np.zeros((512, 512, 12 * vol))
int_2 = np.zeros((512, 512, 12 * (len(lo_list) - vol)))
out_1 = np.zeros((512, 512, 12 * vol))
out_2 = np.zeros((512, 512, 12 * (len(out_list) - vol)))

samp_grid = np.array(np.meshgrid(np.arange(vol_dims[0]), np.arange(vol_dims[1]), np.arange(vol_dims[2])))
samp_grid = np.moveaxis(samp_grid, 0, -1)


# for i in range(vol):
#     hi_vol = np.load(FILE_PATH + 'Hi/' + hi_list[i])
#     lo_vol = np.load(FILE_PATH + 'Lo/' + lo_list[i])
#     out_vol = np.load(OUT_PATH + out_list[i])

#     interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
#                                                           np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
#     int_vol = interpFunc(samp_grid, method='linear')
#     int_vol = np.swapaxes(int_vol, 0, 1)

#     int_1[:, :, (i * 12):((i + 1) * 12)] = int_vol
#     hi_1[:, :, (i * 12):((i + 1) * 12)] = hi_vol
#     out_1[:, :, (i * 12):((i + 1) * 12)] = out_vol
#     print("{}, {}, {}".format(hi_list[i], lo_list[i], out_list[i]))

# nrrd.write(os.path.join(SAVE_PATH, subject + '_1_1_H.nrrd'), hi_1)
# nrrd.write(os.path.join(SAVE_PATH, subject + '_1_1_I.nrrd'), int_1)
# nrrd.write(os.path.join(SAVE_PATH, subject + '_1_1_O.nrrd'), out_1)
# print("SAVED")

for i in range(len(hi_list) - vol):
    hi_vol = np.load(FILE_PATH + 'Hi/' + hi_list[i + vol])
    lo_vol = np.load(FILE_PATH + 'Lo/' + lo_list[i + vol])
    out_vol = np.load(OUT_PATH + out_list[i + vol])

    interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
                                                          np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
    int_vol = interpFunc(samp_grid, method='linear')
    int_vol = np.swapaxes(int_vol, 0, 1)

    int_2[:, :, (i * 12):((i + 1) * 12)] = int_vol
    hi_2[:, :, (i * 12):((i + 1) * 12)] = hi_vol
    out_2[:, :, (i * 12):((i + 1) * 12)] = out_vol
    print("{}, {}, {}".format(hi_list[i + vol], lo_list[i + vol], out_list[i + vol]))

nrrd.write(os.path.join(SAVE_PATH, subject + '_1_2_H.nrrd'), hi_2)
nrrd.write(os.path.join(SAVE_PATH, subject + '_1_2_I.nrrd'), int_2)
nrrd.write(os.path.join(SAVE_PATH, subject + '_1_2_O.nrrd'), out_2)
print("SAVED")
