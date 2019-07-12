from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sciint
import scipy.stats as sci
import skimage.measure as sk


# FILE_PATH = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY_Test/"
FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

image_res = arguments.resolution

vol_dims = [image_res, image_res, 12]
IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/saved_images/" + expt_name + "/"
# IMAGE_SAVE_PATH = FILE_PATH + expt_name + '/'
os.chdir(FILE_PATH)

hi_list = os.listdir('Hi/')
lo_list = os.listdir('Lo/')
hi_list.sort()
lo_list.sort()
out_list = [img[:-5] + 'O.npy' for img in hi_list]

out_test_L2 = []
lo_MSE = []
int_MSE = []
out_MSE = []
lo_pSNR = []
out_pSNR = []
int_pSNR = []
lo_SSIM = []
out_SSIM = []
int_SSIM = []

for idx in range(len(hi_list)):
    hi_vol = np.load('Hi/' + hi_list[idx])
    lo_vol = np.load('Lo/' + lo_list[idx])
    out_vol = np.load(IMAGE_SAVE_PATH + out_list[idx])

    hi_vol = (hi_vol - hi_vol.min()) / (hi_vol.max() - hi_vol.min())
    lo_vol = (lo_vol - lo_vol.min()) / (lo_vol.max() - lo_vol.min())
    out_vol = (out_vol - out_vol.min()) / (out_vol.max() - out_vol.min())

    if hi_list[idx][:-5] != lo_list[idx][:-5] or hi_list[idx][:-5] != out_list[idx][:-5]:
        raise ValueError("Names do not match")
    
    out_test_L2.append(np.sum(np.square(hi_vol - out_vol)))

    samp_grid = np.array(np.meshgrid(np.arange(vol_dims[0]), np.arange(vol_dims[1]), np.arange(vol_dims[2])))
    interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
                                                            np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
    samp_grid = np.moveaxis(samp_grid, 0, -1)
    int_vol = interpFunc(samp_grid, method='linear')
    int_vol = np.float32(np.swapaxes(int_vol, 0, 1))

    lo_MSE.append(sk.compare_mse(hi_vol, lo_vol))
    int_MSE.append(sk.compare_mse(hi_vol, int_vol))
    out_MSE.append(sk.compare_mse(hi_vol, out_vol))
    lo_pSNR.append(sk.compare_psnr(hi_vol, lo_vol))
    int_pSNR.append(sk.compare_psnr(hi_vol, int_vol))
    out_pSNR.append(sk.compare_psnr(hi_vol, out_vol))
    lo_SSIM.append(sk.compare_ssim(hi_vol, lo_vol))
    int_SSIM.append(sk.compare_ssim(hi_vol, int_vol))
    out_SSIM.append(sk.compare_ssim(hi_vol, out_vol))

    print("{:.2f}%".format(idx / len(hi_list) * 100))

print(np.mean(out_test_L2))
MSE_array = np.array([np.array(lo_MSE), np.array(int_MSE), np.array(out_MSE)]).T
pSNR_array = np.array([np.array(lo_pSNR), np.array(int_pSNR), np.array(out_pSNR)]).T
SSIM_array = np.array([np.array(lo_SSIM), np.array(int_SSIM), np.array(out_SSIM)]).T

fig, axs = plt.subplots(1, 3)
axs[0].boxplot(MSE_array)
axs[0].set_xticklabels(['Lo Res', 'Interp', 'Network'])
axs[0].set_title("MSE")
axs[1].boxplot(pSNR_array)
axs[1].set_xticklabels(['Lo Res', 'Interp', 'Network'])
axs[1].set_title("pSNR")
axs[2].boxplot(SSIM_array)
axs[2].set_xticklabels(['Lo Res', 'Interp', 'Network'])
axs[2].set_title("SSIM")

F_MSE, p_MSE = sci.f_oneway(lo_MSE, int_MSE, out_MSE)
print("MSE sig: {} {}".format(F_MSE, p_MSE))
F_pSNR, p_pSNR = sci.f_oneway(lo_pSNR, int_pSNR, out_pSNR)
print("pSNR sig: {} {}".format(F_pSNR, p_pSNR))
F_SSIM, p_SSIM = sci.f_oneway(lo_SSIM, int_SSIM, out_SSIM)
print(F_SSIM, p_SSIM)

plt.show()
