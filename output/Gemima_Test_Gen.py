from argparse import ArgumentParser
import numpy as np
import os
import tensorflow as tf

from Gemima_Utils import UNet, imgLoader, lossMSE


# hi_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "F:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"
hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/Hi/"
lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/Lo/"

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
parser.add_argument('--num_test', '-n', help="Number to test", type=int)
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

model_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_res/models/" + expt_name + "/"
image_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_res/saved_images/" + expt_name + "/"

vol_dims = [size_mb, image_res, image_res, 12, 1]

hi_list = os.listdir(hi_path)
lo_list = os.listdir(lo_path)
hi_list = list(map(lambda img: hi_path + img, hi_list))
lo_list = list(map(lambda img: lo_path + img, lo_list))
assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"

if arguments.num_test == None:
    N = len(hi_list)
else:
    N = arguments.num_test

indices = list(range(0, N))

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

SRNet = UNet(ph_lo)
pred_images = SRNet.output

loss = lossMSE(ph_hi, pred_images)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, model_save_path + expt_name)

    for iter in range(0, N, size_mb):
        hi_mb, lo_mb = imgLoader(hi_list, lo_list, indices[iter:iter+size_mb])
        test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
        output = sess.run(pred_images, feed_dict=test_feed)
        print("Test loss: {}".format(sess.run(loss, feed_dict=test_feed)))

        for idx in range(0, size_mb):
            np.save(image_save_path + hi_list[iter+idx][-26:-5] + 'O', output[idx, :, :, :, 0])
