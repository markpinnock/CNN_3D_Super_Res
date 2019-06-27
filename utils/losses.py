import numpy as np
import scipy.ndimage as scind
import tensorflow as tf

def lossL1(hi_img, pred_img):
    return tf.reduce_sum(tf.abs(hi_img - pred_img), axis=[0, 1, 2, 3])


def lossL2(hi_img, pred_img):
    return tf.reduce_sum(tf.square(hi_img - pred_img), axis=[0, 1, 2, 3])


def regFFT(hi_img, pred_img):
    dims = pred_img.get_shape().as_list()

    reg_val = 0

    for idx in list(range(dims[0])):
        fft_vol = tf.signal.fft(tf.cast(tf.reshape(pred_img[idx, :, :, :, 0]\
                - hi_img[idx, :, :, :, 0], shape=dims[1:4]), dtype=tf.complex64))

        reg_val += tf.reduce_mean(tf.math.abs(fft_vol))

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
