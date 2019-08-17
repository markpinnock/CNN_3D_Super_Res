from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sciint
import skimage.measure as sk


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/"

if not os.path.exists(FILE_PATH):
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"
    # FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/"
    pass

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--volume', '-v', help="Volume number", type=int)
parser.add_argument('--imhist', '-i', help="Image histogram", type=str, nargs='?', const='n', default='n')
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase + '/'
image_res = arguments.resolution

if arguments.volume == None:
    raise ValueError("Must provide volume number")
else:
    vol = arguments.volume

if arguments.imhist == 'y':
    hist_flag = True
else:
    hist_flag = False

vol_dims = [image_res, image_res, 12]

IMAGE_SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/" + phase + expt_name + '/'
# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase + expt_name + "/"
# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/Out/" + expt_name + "/"

if not os.path.exists(IMAGE_SAVE_PATH):
    # IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase + expt_name + "/"
    # IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/Out/" + expt_name + "/"
    pass

hi_list = os.listdir(FILE_PATH + 'Hi/')
lo_list = os.listdir(FILE_PATH + 'Lo/')
hi_list.sort()
lo_list.sort()
output_list = [img[:-5] + 'O.npy' for img in hi_list]

hi_vol = np.load(FILE_PATH + 'Hi/' + hi_list[vol])
lo_vol = np.load(FILE_PATH + 'Lo/' + lo_list[vol])
out_vol = np.load(IMAGE_SAVE_PATH + output_list[vol])

hi_max_val = -10000
hi_min_val = 10000
out_max_val = -10000
out_min_val = 10000

# # # Normalise over subject
# for hi, out in zip(hi_list, output_list):
#     hvol = np.load(FILE_PATH + 'Hi/' + hi)
#     ovol = np.load(IMAGE_SAVE_PATH + out)

#     if hvol.max() > hi_max_val:
#         hi_max_val = hvol.max()
    
#     if hvol.min() < hi_min_val:
#         hi_min_val = hvol.min()

#     if ovol.max() > out_max_val:
#         out_max_val = ovol.max()
    
#     if ovol.min() < out_min_val:
#         out_min_val = ovol.min()


# hi_vol = (hi_vol - hi_vol.min()) / (hi_vol.max() - hi_vol.min())
# lo_vol = (lo_vol - lo_vol.min()) / (lo_vol.max() - lo_vol.min())
# out_vol = (out_vol - out_vol.min()) / (out_vol.max() - out_vol.min())

samp_grid = np.array(np.meshgrid(np.arange(vol_dims[0]), np.arange(vol_dims[1]), np.arange(vol_dims[2])))
interpFunc = sciint.interpolate.RegularGridInterpolator((np.arange(vol_dims[0]), np.arange(vol_dims[1]),
                                                          np.linspace(0, vol_dims[2], 3)), lo_vol[:, :, 2::4])
samp_grid = np.moveaxis(samp_grid, 0, -1)
int_vol = interpFunc(samp_grid, method='linear')
int_vol = np.swapaxes(int_vol, 0, 1)

vol_L2 = np.sum(np.square(hi_vol - out_vol))
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
    lo_L2 = np.sum(np.square(hi_vol[:, :, idx] - lo_vol[:, :, idx]))
    out_L2 = np.sum(np.square(hi_vol[:, :, idx] - out_vol[:, :, idx]))
    int_L2 = np.sum(np.square(hi_vol[:, :, idx] - int_vol[:, :, idx]))

    print(hi_vol[:, :, idx].max(), hi_vol[:, :, idx].min())
    print(lo_vol[:, :, idx].max(), lo_vol[:, :, idx].min())
    print(out_vol[:, :, idx].max(), out_vol[:, :, idx].min())
    print("\n")
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Axial: subject {}, slice {}, volume L2 {:.2f} ({:.2f} per image)'.format(vol, idx, vol_L2, vol_L2 / 12))

    if hist_flag == False:
        axs[0, 0].imshow(hi_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[0, 0].set_title('Hi res')
        axs[0, 0].axis('off')
    
        axs[1, 0].imshow(lo_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[1, 0].set_title('Lo res (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(lo_L2, lo_pSNR, lo_SSIM))
        axs[1, 0].axis('off')
        
        axs[0, 1].imshow(int_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[0, 1].set_title('Interp (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(int_L2, int_pSNR, int_SSIM))
        axs[0, 1].axis('off')
        
        axs[1, 1].imshow(hi_vol[:, :, idx].T - int_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[1, 1].set_title('Difference')
        axs[1, 1].axis('off')
        
        axs[0, 2].imshow(out_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[0, 2].set_title('Network (Slice L2, {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(out_L2, out_pSNR, out_SSIM))
        axs[0, 2].axis('off')
        
        axs[1, 2].imshow(hi_vol[:, :, idx].T - out_vol[:, :, idx].T, cmap='gray', origin='lower')
        axs[1, 2].set_title('Difference')
        axs[1, 2].axis('off')
    
    else:
        axs[0, 0].hist(hi_vol[:, :, idx].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 0].set_title('Hi res')
    
        axs[1, 0].hist(lo_vol[:, :, idx].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[1, 0].set_title('Lo res (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(lo_L2, lo_pSNR, lo_SSIM))
        
        axs[0, 1].hist(int_vol[:, :, idx].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 1].set_title('Interp (Slice L2 {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(int_L2, int_pSNR, int_SSIM))
        
        H_int, _, _ = np.histogram2d(hi_vol[:, :, idx].ravel(), int_vol[:, :, idx].ravel(), bins=50)

        axs[1, 1].imshow(H_int.T, cmap='hot')
        axs[1, 1].set_title('Difference')

        H_out, _, _ = np.histogram2d(hi_vol[:, :, idx].ravel(), out_vol[:, :, idx].ravel(), bins=50)
        
        axs[0, 2].hist(out_vol[:, :, idx].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 2].set_title('Network (Slice L2, {:.2f}, pSNR {:.2f}, SSIM {:.2f})'.format(out_L2, out_pSNR, out_SSIM))
        
        axs[1, 2].imshow(H_out.T, cmap='hot')
        axs[1, 2].set_title('Difference')
    
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

for idx in range(0, vol_dims[0], 16):
    lo_L2 = np.sum(np.square(hi_vol[:, idx, :] - lo_vol[:, idx, :]))
    out_L2 = np.sum(np.square(hi_vol[:, idx, :] - out_vol[:, idx, :]))
    int_L2 = np.sum(np.square(hi_vol[:, idx, :] - int_vol[:, idx, :]))
    
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Coronal: subject {}, slice {}, volume L2 {:.2f} ({:.2f} per image)'.format(vol, idx, vol_L2, vol_L2 / 12))
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
    lo_L2 = np.sum(np.square(hi_vol[idx, :, :] - lo_vol[idx, :, :]))
    out_L2 = np.sum(np.square(hi_vol[idx, :, :] - out_vol[idx, :, :]))
    int_L2 = np.sum(np.square(hi_vol[idx, :, :] - int_vol[idx, :, :]))

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Saggital: subject {}, slice {}, volume L2 {:.2f} ({:.2f} per image)'.format(vol, idx, vol_L2, vol_L2 / 12))
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
