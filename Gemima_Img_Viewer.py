from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.interpolate as sciint
import skimage.measure as sk


parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--volume', '-v', help="Volume number", type=int)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

image_res = arguments.resolution

if arguments.subject == None:
    raise ValueError("Must provide volume number")
else:
    vol = arguments.minibatch_size


random.seed(10)
num_test = 16
vol_dims = [image_res, image_res, 12]

# hi_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"
hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Hi/"
lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Lo/"
image_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_res/saved_images/" + expt_name

hi_list = os.listdir(hi_path)
lo_list = os.listdir(lo_path)
temp_list = list(zip(hi_list, lo_list))
random.shuffle(temp_list)
hi_list, lo_list = zip(*temp_list)
test_hi_list = hi_list[0:num_test]
test_lo_list = lo_list[0:num_test]

output_list = [img[-26:-5] + 'O.npy' for img in test_hi_list]

hi_vol = np.load(hi_path + test_hi_list[vol])
lo_vol = np.load(lo_path + test_lo_list[vol])
out_vol = np.load(image_save_path + output_list[vol])
print(np.min(hi_vol, axis=(0, 1, 2)), np.max(hi_vol, axis=(0, 1, 2)))
print(np.min(lo_vol, axis=(0, 1, 2)), np.max(lo_vol, axis=(0, 1, 2)))
print(np.min(out_vol, axis=(0, 1, 2)), np.max(out_vol, axis=(0, 1, 2)))

samp_grid = np.array(np.meshgrid(np.arange(vol_dims[0]), np.arange(vol_dims[1]), np.arange(vol_dims[2])))
interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
                                                          np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
samp_grid = np.moveaxis(samp_grid, 0, -1)
int_vol = interpFunc(samp_grid, method='linear')
int_vol = np.swapaxes(int_vol, 0, 1)

for idx in range(0, vol_dims[2]):
    # lo_MSE = sk.compare_mse(hi_vol[:, :, idx], lo_vol[:, :, idx])
    # out_MSE = sk.compare_mse(hi_vol[:, :, idx], out_vol[:, :, idx])
    # int_MSE = sk.compare_mse(hi_vol[:, :, idx], int_vol[:, :, idx])
    lo_pSNR = sk.compare_psnr(hi_vol[:, :, idx], lo_vol[:, :, idx])
    out_pSNR = sk.compare_psnr(hi_vol[:, :, idx], out_vol[:, :, idx])
    int_pSNR = sk.compare_psnr(hi_vol[:, :, idx], int_vol[:, :, idx])
    lo_SSIM = sk.compare_ssim(hi_vol[:, :, idx], lo_vol[:, :, idx])
    out_SSIM = sk.compare_ssim(hi_vol[:, :, idx], out_vol[:, :, idx])
    int_SSIM = sk.compare_ssim(hi_vol[:, :, idx], int_vol[:, :, idx])
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Axial: subject {}, slice {}'.format(vol, idx))
    axs[0, 0].imshow(hi_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[1, 0].imshow(lo_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Lo res (pSNR {:.2f}, SSIM {:.2f})'.format(lo_pSNR, lo_SSIM))
    axs[1, 0].axis('off')
    
    axs[0, 1].imshow(int_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Interp (pSNR {:.2f}, SSIM {:.2f})'.format(int_pSNR, int_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 1].imshow(hi_vol[:, :, idx].T - int_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[0, 2].imshow(out_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 2].set_title('Network (pSNR {:.2f}, SSIM {:.2f})'.format(out_pSNR, out_SSIM))
    axs[0, 2].axis('off')
    
    axs[1, 2].imshow(hi_vol[:, :, idx].T - out_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 2].set_title('Difference')
    axs[1, 2].axis('off')
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.show()

for idx in range(0, vol_dims[0], 16):
    # lo_MSE = sk.compare_mse(hi_vol[:, idx, :], lo_vol[:, idx, :])
    # out_MSE = sk.compare_mse(hi_vol[:, idx, :], out_vol[:, idx, :])
    # int_MSE = sk.compare_mse(hi_vol[:, idx, :], int_vol[:, idx, :])
    lo_pSNR = sk.compare_psnr(hi_vol[:, idx, :], lo_vol[:, idx, :])
    out_pSNR = sk.compare_psnr(hi_vol[:, idx, :], out_vol[:, idx, :])
    int_pSNR = sk.compare_psnr(hi_vol[:, idx, :], int_vol[:, idx, :])
    lo_SSIM = sk.compare_ssim(hi_vol[:, idx, :], lo_vol[:, idx, :])
    out_SSIM = sk.compare_ssim(hi_vol[:, idx, :], out_vol[:, idx, :])
    int_SSIM = sk.compare_ssim(hi_vol[:, idx, :], int_vol[:, idx, :])
    
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Coronal: subject {}, slice {}'.format(vol, idx))
    axs[0, 0].imshow(hi_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(lo_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Lo res (pSNR {:.2f}, SSIM {:.2f})'.format(lo_pSNR, lo_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(int_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Interp (pSNR {:.2f}, SSIM {:.2f})'.format(int_pSNR, int_SSIM))
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(hi_vol[100:400, idx, :].T - int_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[2, 0].imshow(out_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[2, 0].set_title('Network (pSNR {:.2f}, SSIM {:.2f})'.format(out_pSNR, out_SSIM))
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(hi_vol[100:400, idx, :].T - out_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[2, 1].set_title('Difference')
    axs[2, 1].axis('off')
    
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

for idx in range(0, vol_dims[1], 16):
    # lo_MSE = sk.compare_mse(hi_vol[idx, :, :], lo_vol[idx, :, :])
    # out_MSE = sk.compare_mse(hi_vol[idx, :, :], out_vol[idx, :, :])
    # int_MSE = sk.compare_mse(hi_vol[idx, :, :], int_vol[idx, :, :])
    lo_pSNR = sk.compare_psnr(hi_vol[idx, :, :], lo_vol[idx, :, :])
    out_pSNR = sk.compare_psnr(hi_vol[idx, :, :], out_vol[idx, :, :])
    int_pSNR = sk.compare_psnr(hi_vol[idx, :, :], int_vol[idx, :, :])
    lo_SSIM = sk.compare_ssim(hi_vol[idx, :, :], lo_vol[idx, :, :])
    out_SSIM = sk.compare_ssim(hi_vol[idx, :, :], out_vol[idx, :, :])
    int_SSIM = sk.compare_ssim(hi_vol[idx, :, :], int_vol[idx, :, :])

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Saggital: subject {}, slice {}'.format(vol, idx))
    axs[0, 0].imshow(hi_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(lo_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Lo res (pSNR {:.2f}, SSIM {:.2f})'.format(lo_pSNR, lo_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(int_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Interp (pSNR {:.2f}, SSIM {:.2f})'.format(int_pSNR, int_SSIM))
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(hi_vol[idx, 100:400, :].T - int_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[2, 0].imshow(out_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[2, 0].set_title('Network (pSNR {:.2f}, SSIM {:.2f})'.format(out_pSNR, out_SSIM))
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(hi_vol[idx, 100:400, :].T - out_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[2, 1].set_title('Difference')
    axs[2, 1].axis('off')

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())   
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
