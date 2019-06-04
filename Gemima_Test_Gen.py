from argparse import ArgumentParser
import numpy as np
import random
import os
import tensorflow as tf

from Gemima_Utils import UNet, imgLoader


# hi_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"
hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Hi/"
lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Lo/"
model_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_res/models/"

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

image_res = arguments.resolution

if arguments.minibatch_size == None:
    raise ValueError("Must provide minibatch size")
else:
    size_mb = arguments.minibatch_size

image_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_res/saved_images/" + expt_name + "/"

num_test = 16
vol_dims = [size_mb, image_res, image_res, 12, 1]
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

SRNet = UNet(ph_lo)
pred_images = SRNet.output

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, model_save_path + expt_name)

    for iter in range(0, num_test, size_mb):
        hi_mb, lo_mb = imgLoader(test_hi_list, test_lo_list, indices[iter:iter+size_mb])
        test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
        output = sess.run(pred_images, feed_dict=test_feed)

        for idx in range(0, size_mb):
            np.save(image_save_path + test_hi_list[iter+idx][-26:-5] + 'O', output[idx, :, :, :, 0])
