import numpy as np
import scipy.ndimage as scind
import skimage.measure as sk
import tensorflow as tf


def calcDice(vol_A, vol_B):
    numer = 2 * np.sum((vol_A * vol_B))
    denom = np.sum(vol_A) + np.sum(vol_B) + 1e-6
    return numer / denom


def lossL1(hi_img, pred_img):
    return tf.reduce_sum(tf.reduce_mean(tf.abs(hi_img - pred_img), axis=[1, 2, 3, 4]), axis=None)


def lossL2(hi_img, pred_img):
    return tf.reduce_sum(tf.reduce_mean(tf.square(hi_img - pred_img), axis=[1, 2, 3, 4]), axis=None)


def calcPSNR(hi_img, pred_img):
    dims = pred_img.shape
    val_pSNR = 0

    for idx in list(range(dims[0])):
        val_pSNR += sk.compare_psnr(hi_img[idx, :, :, :, 0], pred_img[idx, :, :, :, 0])

    return val_pSNR


def calcSSIM(hi_img, pred_img):
    dims = pred_img.shape
    val_SSIM = 0

    for idx in list(range(dims[0])):
        val_SSIM += sk.compare_ssim(hi_img[idx, :, :, :, 0], pred_img[idx, :, :, :, 0])

    return val_SSIM


def regFFT(hi_img, pred_img):
    dims = pred_img.get_shape().as_list()

    reg_val = 0

    for idx in list(range(dims[0])):
        fft_vol = tf.signal.fft(tf.cast(tf.reshape(pred_img[idx, :, :, :, 0]\
                - hi_img[idx, :, :, :, 0], shape=[-1]), dtype=tf.complex64))

        k_vals = tf.linspace(0.0, 2.0, dims[1] * dims[2] * dims[3])

        fft_vals = fft_vol * tf.cast(k_vals, dtype=tf.complex64)
        
        # reg_val += tf.contrib.distributions.percentile(tf.math.abs(fft_vals), q=95, axis=None)
        reg_val += tf.reduce_sum(tf.reduce_mean(tf.math.abs(fft_vals), axis=[1, 2, 3, 4]), axis=None)

    return reg_val


def reg3DFFT(hi_img, pred_img):
    dims = pred_img.get_shape().as_list()

    reg_val = 0

    for idx in list(range(dims[0])):
        fft_vol = tf.signal.fft3d(tf.cast(pred_img[idx, :, :, :, 0]\
                - hi_img[idx, :, :, :, 0], dtype=tf.complex64))

        N = dims[1] * dims[2] * dims[3]
        flat_fft = tf.reshape(fft_vol, shape=[-1])
        k_vals = tf.linspace(0.0, 2.0, N)

        fft_vals = flat_fft[0:int(N/2)] * tf.cast(k_vals[0:int(N/2)], dtype=tf.complex64)
        
        # reg_val += tf.contrib.distributions.percentile(tf.math.abs(fft_vals), q=95, axis=None)
        reg_val += tf.reduce_sum(tf.reduce_mean(tf.math.abs(fft_vals), axis=[1, 2, 3, 4]), axis=None)

    return reg_val


def regLaplace(hi_img, pred_img):
    dims = pred_img.get_shape().as_list()
    lap_filt_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    lap_filt_2 = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
    lap_filt_3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    lap_filt_3d = np.stack((lap_filt_1, lap_filt_2, lap_filt_3), axis=2)

    lap_filt_3d = tf.cast(tf.convert_to_tensor(lap_filt_3d), dtype=tf.float32)
    lap_filt_3d = lap_filt_3d[:, :, :, tf.newaxis, tf.newaxis]

    filt_vols = tf.nn.conv3d(hi_img - pred_img, lap_filt_3d,\
            strides=[1, 1, 1, 1, 1], padding='SAME')

    filt_vols = tf.square(filt_vols)
#     reg_val = tf.reduce_sum(tf.contrib.distributions.percentile(
#             filt_vols, q=95, axis=[1, 2, 3, 4]))
    reg_val = tf.reduce_sum(filt_vols)

    return reg_val
