from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sciint
import skimage.measure as sk


# hi_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"
FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"

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

if arguments.volume == None:
    raise ValueError("Must provide volume number")
else:
    vol = arguments.volume

vol_dims = [image_res, image_res, 12]
IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/saved_images/" + expt_name + "/"

hi_list = os.listdir(FILE_PATH + 'Hi/')
lo_list = os.listdir(FILE_PATH + 'Lo/')
output_list = [img[:-5] + 'O.npy' for img in hi_list]

hi_vol = np.load(FILE_PATH + '/Hi' + hi_list[vol])
lo_vol = np.load(FILE_PATH + 'Lo/' + lo_list[vol])
out_vol = np.load(IMAGE_SAVE_PATH + output_list[vol])

# # Normalise
# hi_vol = (hi_vol - hi_vol.min()) / (hi_vol.max() - hi_vol.min())
# lo_vol = (lo_vol - lo_vol.min()) / (lo_vol.max() - lo_vol.min())
# out_vol = (out_vol - out_vol.min()) / (out_vol.max() - out_vol.min())

samp_grid = np.array(np.meshgrid(np.arange(vol_dims[0]), np.arange(vol_dims[1]), np.arange(vol_dims[2])))
interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
                                                          np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
samp_grid = np.moveaxis(samp_grid, 0, -1)
int_vol = interpFunc(samp_grid, method='linear')
int_vol = np.swapaxes(int_vol, 0, 1)

vol_L2 = np.sum(np.square(hi_vol - lo_vol))
lo_MSE = sk.compare_mse(hi_vol, lo_vol)
out_MSE = sk.compare_mse(hi_vol, out_vol)
int_MSE = sk.compare_mse(hi_vol, int_vol)
# lo_L2 = np.sum(np.square(hi_vol - lo_vol))
# out_L2 = np.sum(np.square(hi_vol - out_vol))
# int_L2 = np.sum(np.square(hi_vol - int_vol))
lo_pSNR = sk.compare_psnr(hi_vol, lo_vol)
out_pSNR = sk.compare_psnr(hi_vol, out_vol)
int_pSNR = sk.compare_psnr(hi_vol, int_vol)
lo_SSIM = sk.compare_ssim(hi_vol, lo_vol)
out_SSIM = sk.compare_ssim(hi_vol, out_vol)
int_SSIM = sk.compare_ssim(hi_vol, int_vol)

for idx in range(0, vol_dims[2]):
    # # lo_MSE = sk.compare_mse(hi_vol, lo_vol)
    # # out_MSE = sk.compare_mse(hi_vol, out_vol)
    # # int_MSE = sk.compare_mse(hi_vol, int_vol)
    # lo_L2 = np.sum(np.square(hi_vol - lo_vol))
    # out_L2 = np.sum(np.square(hi_vol - out_vol))
    # int_L2 = np.sum(np.square(hi_vol - int_vol))
    # lo_pSNR = sk.compare_psnr(hi_vol, lo_vol)
    # out_pSNR = sk.compare_psnr(hi_vol, out_vol)
    # int_pSNR = sk.compare_psnr(hi_vol, int_vol)
    # lo_SSIM = sk.compare_ssim(hi_vol, lo_vol)
    # out_SSIM = sk.compare_ssim(hi_vol, out_vol)
    # int_SSIM = sk.compare_ssim(hi_vol, int_vol)

    print(hi_vol[:, :, idx].max(), hi_vol[:, :, idx].min())
    print(lo_vol[:, :, idx].max(), lo_vol[:, :, idx].min())
    print(out_vol[:, :, idx].max(), out_vol[:, :, idx].min())
    print("\n")
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Axial: subject {}, slice {}, volume L2 {:.2f}'.format(vol, idx, vol_L2))
    axs[0, 0].imshow(hi_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[1, 0].imshow(lo_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Lo res (MSE {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(lo_MSE, lo_pSNR, lo_SSIM))
    axs[1, 0].axis('off')
    
    axs[0, 1].imshow(int_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Interp (MSE {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(int_MSE, int_pSNR, int_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 1].imshow(hi_vol[:, :, idx].T - int_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[0, 2].imshow(out_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[0, 2].set_title('Network (MSE, {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(out_MSE, out_pSNR, out_SSIM))
    axs[0, 2].axis('off')
    
    axs[1, 2].imshow(hi_vol[:, :, idx].T - out_vol[:, :, idx].T, cmap='gray', origin='lower')
    axs[1, 2].set_title('Difference')
    axs[1, 2].axis('off')
    
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

for idx in range(0, vol_dims[0], 16):
    # # lo_MSE = sk.compare_mse(hi_vol[:, idx, :], lo_vol[:, idx, :])
    # # out_MSE = sk.compare_mse(hi_vol[:, idx, :], out_vol[:, idx, :])
    # # int_MSE = sk.compare_mse(hi_vol[:, idx, :], int_vol[:, idx, :])
    # lo_L2 = np.sum(np.square(hi_vol[:, idx, :] - lo_vol[:, idx, :]))
    # out_L2 = np.sum(np.square(hi_vol[:, idx, :] - out_vol[:, idx, :]))
    # int_L2 = np.sum(np.square(hi_vol[:, idx, :] - int_vol[:, idx, :]))
    # lo_pSNR = sk.compare_psnr(hi_vol[:, idx, :], lo_vol[:, idx, :])
    # out_pSNR = sk.compare_psnr(hi_vol[:, idx, :], out_vol[:, idx, :])
    # int_pSNR = sk.compare_psnr(hi_vol[:, idx, :], int_vol[:, idx, :])
    # lo_SSIM = sk.compare_ssim(hi_vol[:, idx, :], lo_vol[:, idx, :])
    # out_SSIM = sk.compare_ssim(hi_vol[:, idx, :], out_vol[:, idx, :])
    # int_SSIM = sk.compare_ssim(hi_vol[:, idx, :], int_vol[:, idx, :])
    
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Coronal: subject {}, slice {}, volume L2 {:.2f}'.format(vol, idx, vol_L2))
    axs[0, 0].imshow(hi_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(lo_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Lo res (Slice L2, {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(lo_L2, lo_pSNR, lo_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(int_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Interp (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(int_L2, int_pSNR, int_SSIM))
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(hi_vol[100:400, idx, :].T - int_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[2, 0].imshow(out_vol[100:400, idx, :].T, cmap='gray', origin='lower')
    axs[2, 0].set_title('Network (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(out_L2, out_pSNR, out_SSIM))
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
    # # lo_MSE = sk.compare_mse(hi_vol[idx, :, :], lo_vol[idx, :, :])
    # # out_MSE = sk.compare_mse(hi_vol[idx, :, :], out_vol[idx, :, :])
    # # int_MSE = sk.compare_mse(hi_vol[idx, :, :], int_vol[idx, :, :])
    # lo_L2 = np.sum(np.square(hi_vol[idx, :, :] - lo_vol[idx, :, :]))
    # out_L2 = np.sum(np.square(hi_vol[idx, :, :] - out_vol[idx, :, :]))
    # int_L2 = np.sum(np.square(hi_vol[idx, :, :] - int_vol[idx, :, :]))
    # lo_pSNR = sk.compare_psnr(hi_vol[idx, :, :], lo_vol[idx, :, :])
    # out_pSNR = sk.compare_psnr(hi_vol[idx, :, :], out_vol[idx, :, :])
    # int_pSNR = sk.compare_psnr(hi_vol[idx, :, :], int_vol[idx, :, :])
    # lo_SSIM = sk.compare_ssim(hi_vol[idx, :, :], lo_vol[idx, :, :])
    # out_SSIM = sk.compare_ssim(hi_vol[idx, :, :], out_vol[idx, :, :])
    # int_SSIM = sk.compare_ssim(hi_vol[idx, :, :], int_vol[idx, :, :])

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Saggital: subject {}, slice {}, volume L2 {:.2f}'.format(vol, idx, vol_L2))
    axs[0, 0].imshow(hi_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[0, 0].set_title('Hi res')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(lo_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[0, 1].set_title('Lo res (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(lo_L2, lo_pSNR, lo_SSIM))
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(int_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[1, 0].set_title('Interp (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(int_L2, int_pSNR, int_SSIM))
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(hi_vol[idx, 100:400, :].T - int_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[1, 1].set_title('Difference')
    axs[1, 1].axis('off')
    
    axs[2, 0].imshow(out_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[2, 0].set_title('Network (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(out_L2, out_pSNR, out_SSIM))
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(hi_vol[idx, 100:400, :].T - out_vol[idx, 100:400, :].T, cmap='gray', origin='lower')
    axs[2, 1].set_title('Difference')
    axs[2, 1].axis('off')

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())   
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
