import numpy as np
import random
import os
import tensorflow as tf

from Gemima_Utils import UNet, imgLoader


# hi_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"

name = 'test1HPC'
hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Hi/"
lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Lo/"
model_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_res/models/"
image_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_res/saved_images/" + name + "/"

size_mb = 16
num_test = 16
vol_dims = [size_mb, 128, 128, 12, 1]
random.seed(10)

hi_list = os.listdir(hi_path)
lo_list = os.listdir(lo_path)
temp_list = list(zip(hi_list, lo_list))
random.shuffle(temp_list)
hi_list, lo_list = zip(*temp_list)
assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"

hi_list = list(map(lambda img: hi_path + img, hi_list))
lo_list = list(map(lambda img: lo_path + img, lo_list))

test_hi_list = hi_list[0:num_test]
test_lo_list = lo_list[0:num_test]

indices = list(range(0, len(test_hi_list)))

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

GemNet = UNet(ph_lo)
pred_images = GemNet.output

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, model_save_path + name)

    for iter in range(0, num_test, size_mb):
        hi_mb, lo_mb = imgLoader(test_hi_list, test_lo_list, indices[iter:iter+size_mb])
        test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
        output = sess.run(pred_images, feed_dict=test_feed)

        for idx in range(0, size_mb):
            np.save(image_save_path + test_hi_list[iter+idx][-26:-5] + 'O', output[idx, :, :, :, 0])
