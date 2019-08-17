import nrrd
import numpy as np
import os
import skimage.measure as sk


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/"
METRIC_SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/Metric_Test/"

hi_list = [img for img in os.listdir(FILE_PATH + 'Hi/') if '.seg' not in img] 
lo_list = [img for img in os.listdir(FILE_PATH + 'Lo/') if '.seg' not in img]
int_list = [img for img in os.listdir(FILE_PATH + 'Int/') if '.seg' not in img]
hi_list.sort()
lo_list.sort()
int_list.sort()

lo_MSE = []
int_MSE = []
lo_pSNR = []
int_pSNR = []
lo_SSIM = []
int_SSIM = []

for idx in range(len(lo_list)):
    lo_vol, _ = nrrd.read(FILE_PATH + 'Lo/' + lo_list[idx])
    int_vol, _ = nrrd.read(FILE_PATH + 'Int/' + int_list[idx])
    hi_vol, _ = nrrd.read(FILE_PATH + 'Hi/' + hi_list[idx])
    dims = int_vol.shape

    for sub in range(0, dims[2], 12):
        lo_sub_vol = lo_vol[sub:sub + 12]
        int_sub_vol = int_vol[sub:sub + 12]
        hi_sub_vol = hi_vol[sub:sub + 12]
        # lo_sub_vol = (lo_sub_vol - lo_sub_vol.min()) / (lo_sub_vol.max() - lo_sub_vol.min())
        # int_sub_vol = (int_sub_vol - int_sub_vol.min()) / (int_sub_vol.max() - int_sub_vol.min())
        # hi_sub_vol = (hi_sub_vol - hi_sub_vol.min()) / (hi_sub_vol.max() - hi_sub_vol.min())

        lo_MSE.append(sk.compare_mse(hi_sub_vol, lo_sub_vol))
        int_MSE.append(sk.compare_mse(hi_sub_vol, int_sub_vol))
        lo_pSNR.append(sk.compare_psnr(hi_sub_vol, lo_sub_vol))
        int_pSNR.append(sk.compare_psnr(hi_sub_vol, int_sub_vol))
        lo_SSIM.append(sk.compare_ssim(hi_sub_vol, lo_sub_vol))
        int_SSIM.append(sk.compare_ssim(hi_sub_vol, int_sub_vol))
    
    print(idx / len(lo_list) * 100)

np.save(METRIC_SAVE_PATH + 'lo_MSE', np.array(lo_MSE))
np.save(METRIC_SAVE_PATH + 'lo_pSNR', np.array(lo_pSNR))
np.save(METRIC_SAVE_PATH + 'lo_SSIM', np.array(lo_SSIM))

np.save(METRIC_SAVE_PATH + 'int_MSE', np.array(int_MSE))
np.save(METRIC_SAVE_PATH + 'int_pSNR', np.array(int_pSNR))
np.save(METRIC_SAVE_PATH + 'int_SSIM', np.array(int_SSIM))