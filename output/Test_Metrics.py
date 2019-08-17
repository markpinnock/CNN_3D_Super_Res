from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sciint
import scipy.stats as sci
import skimage.measure as sk


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"

if not os.path.exists(FILE_PATH):
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"
    pass

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help="Expt phase", type=str, nargs='?', const='3', default='3')
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase + '/'
image_res = arguments.resolution

vol_dims = [image_res, image_res, 12]

IMAGE_SAVE_PATH = FILE_PATH + phase + expt_name + '/'
# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase + expt_name + "/"

if not os.path.exists(IMAGE_SAVE_PATH):
    # IMAGE_SAVE_PATH "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase + expt_name + "/"
    pass

METRIC_SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/Metric_Test/" + phase
os.chdir(FILE_PATH)

hi_list = os.listdir('Hi/')
hi_list.sort()
out_list = [img[:-5] + 'O.npy' for img in hi_list]

out_test_L2 = []
out_MSE = []
out_pSNR = []
out_SSIM = []

for idx in range(len(hi_list)):
    hi_vol = np.load('Hi/' + hi_list[idx])
    out_vol = np.load(IMAGE_SAVE_PATH + out_list[idx])
    
    out_test_L2.append(np.sum(np.square(hi_vol - out_vol)))

for idx in range(len(hi_list)):
    hi_vol = np.load('Hi/' + hi_list[idx])
    out_vol = np.load(IMAGE_SAVE_PATH + out_list[idx])

    # hi_vol = (hi_vol - hi_vol.min()) / (hi_vol.max() - hi_vol.min())
    # out_vol = (out_vol - out_vol.min()) / (out_vol.max() - out_vol.min())

    if hi_list[idx][:-5] != out_list[idx][:-5]:
        raise ValueError("Names do not match")

    out_MSE.append(sk.compare_mse(hi_vol, out_vol))
    out_pSNR.append(sk.compare_psnr(hi_vol, out_vol))
    out_SSIM.append(sk.compare_ssim(hi_vol, out_vol))

    # print("{:.2f}%".format(idx / len(hi_list) * 100))

print("Test loss: {:.2f}".format(np.mean(out_test_L2)))
# MSE_array = np.array([np.array(lo_MSE), np.array(int_MSE), np.array(out_MSE)]).T
# pSNR_array = np.array([np.array(lo_pSNR), np.array(int_pSNR), np.array(out_pSNR)]).T
# SSIM_array = np.array([np.array(lo_SSIM), np.array(int_SSIM), np.array(out_SSIM)]).T
# print("MSE: {:.2f}".format(np.mean(out_MSE)))
# print("pSNR: {:.2f}".format(np.mean(out_pSNR)))
# print("SSIM: {:.2f}".format(np.mean(out_SSIM)))

# fig, axs = plt.subplots(1, 3)
# axs[0].boxplot(MSE_array)
# axs[0].set_xticklabels(['Lo Res', 'Interp', 'Network'])
# axs[0].set_title("MSE")
# axs[1].boxplot(pSNR_array)
# axs[1].set_xticklabels(['Lo Res', 'Interp', 'Network'])
# axs[1].set_title("pSNR")
# axs[2].boxplot(SSIM_array)
# axs[2].set_xticklabels(['Lo Res', 'Interp', 'Network'])
# axs[2].set_title("SSIM")

# F_MSE, p_MSE = sci.f_oneway(lo_MSE, int_MSE, out_MSE)
# print("MSE sig: F = {:.2f}, p = {:.2f}".format(F_MSE, p_MSE))
# F_pSNR, p_pSNR = sci.f_oneway(lo_pSNR, int_pSNR, out_pSNR)
# print("pSNR sig: F = {:.2f}, p = {:.2f}".format(F_pSNR, p_pSNR))
# F_SSIM, p_SSIM = sci.f_oneway(lo_SSIM, int_SSIM, out_SSIM)
# print("SSIM sig: F = {:.2f}, p = {:.2f}".format(F_SSIM, p_SSIM))

np.save(METRIC_SAVE_PATH + expt_name + '/' + expt_name + '_MSE', np.array(out_MSE))
np.save(METRIC_SAVE_PATH + expt_name + '/' + expt_name + '_pSNR', np.array(out_pSNR))
np.save(METRIC_SAVE_PATH + expt_name + '/' + expt_name + '_SSIM', np.array(out_SSIM))

# np.save(METRIC_SAVE_PATH + 'lo_MSE', np.array(lo_MSE))
# np.save(METRIC_SAVE_PATH + 'lo_pSNR', np.array(lo_pSNR))
# np.save(METRIC_SAVE_PATH + 'lo_SSIM', np.array(lo_SSIM))

# np.save(METRIC_SAVE_PATH + 'int_MSE', np.array(int_MSE))
# np.save(METRIC_SAVE_PATH + 'int_pSNR', np.array(int_pSNR))
# np.save(METRIC_SAVE_PATH + 'int_SSIM', np.array(int_SSIM))

# plt.show()
