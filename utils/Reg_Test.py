import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as scind
import tensorflow as tf

from functions import imgLoader
from losses import regFFT, regLaplace

# FILE_PATH = "F:/PhD/Super_Res_Data/Toshiba_Vols/NII_Test/"
FILE_PATH = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY_Train/"
HI_PATH = FILE_PATH + 'Hi/'
LO_PATH = FILE_PATH + 'Lo/'

os.chdir(FILE_PATH)

hi_list = os.listdir(HI_PATH)
lo_list = os.listdir(LO_PATH)

# hvol = np.float32(nib.load(FILE_PATH + "UCLH_17138405_1_2_H.nii").get_fdata())
# lvol = np.float32(nib.load(FILE_PATH + "UCLH_17138405_1_2_L.nii").get_fdata())
# ivol = np.float32(nib.load(FILE_PATH + "UCLH_17138405_1_2_I.nii").get_fdata())

ph_hi = tf.placeholder(tf.float32, [4, 512, 512, 12, 1])
ph_lo = tf.placeholder(tf.float32, [4, 512, 512, 12, 1])

with tf.Session() as sess:

    hi_mb, lo_mb = imgLoader(hi_list, lo_list, [160, 161, 162, 163])
    train_feed = {ph_hi:hi_mb, ph_lo:lo_mb}
    num = 0
    non_tf_hi = hi_mb[num, :, :, :, 0]
    non_tf_lo = lo_mb[num, :, :, :, 0]

    fig, axs = plt.subplots(1, 3)
    # axs[0, 0].imshow(non_tf_hi[:, :, 6].T, origin='lower', cmap='gray')
    # axs[0, 0].axis('off')
    # axs[0, 1].imshow(non_tf_lo[:, :, 6].T, origin='lower', cmap='gray')
    # axs[0, 1].axis('off')

    hi_lap = scind.filters.laplace(non_tf_hi)
    lo_lap = scind.filters.laplace(non_tf_hi - non_tf_lo)

    # axs[1, 0].imshow(hi_lap[:, :, 6].T, origin='lower', cmap='hot')
    # axs[1, 0].axis('off')
    # axs[1, 1].imshow(lo_lap[:, :, 6].T, origin='lower', cmap='hot')
    # axs[1, 1].axis('off')

    reg_val = sess.run(regLaplace(ph_hi, ph_lo), feed_dict=train_feed)
    print(reg_val, np.quantile(lo_lap**2, 0.95))
    axs[0].imshow(hi_mb[0, :, :, 1, 0].T, origin='lower', cmap='hot')
    axs[0].axis('off')
    axs[1].imshow(hi_lap[:, :, 1].T, origin='lower', cmap='hot')
    # axs[2].imshow(filt_vol[0, :, :, 1, 0].T, origin='lower', cmap='hot')
    axs[2].axis('off')

    plt.show()

# hi_lap = np.zeros((512, 512, 12))
# lo_lap = np.zeros((512, 512, 12))
# int_lap = np.zeros((512, 512, 12))

# lap_filt_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# lap_filt_2 = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
# lap_filt_3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# lap_filt_3d = np.stack((lap_filt_1, lap_filt_2, lap_filt_3), axis=2)


# for i in range(20):

#     # hi_lap = scind.filters.convolve(hi_mb[i, :, :, :], lap_filt_3d)
#     lo_lap = scind.filters.convolve(lo_mb[i, :, :, :] - hi_mb[i, :, :, :], lap_filt_3d)
#     # hi_lap = scind.filters.laplace(hvol)
#     # lo_lap = scind.filters.laplace(lvol)
#     # int_lap = scind.filters.laplace(ivol)
#     fft_img  = np.fft.fft(lo_mb[i, :, :, :].ravel() - hi_mb[i, :, :, :].ravel())
#     print("Laplacian Filter:", np.quantile(hi_lap**2, 0.95), np.quantile(lo_lap**2, 0.95))
#     print("FFT:", np.mean(np.abs(fft_img)))