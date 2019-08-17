from argparse import ArgumentParser
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')

from utils.UNet import UNet
from utils.functions import imgLoader
from utils.losses import lossL2


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/"

if not os.path.exists(FILE_PATH):
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/"
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/real_test_imgs/"
    pass

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=8, default=8)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase + '/'
image_res = arguments.resolution

if arguments.minibatch_size == None:
    raise ValueError("Must provide minibatch size")
else:
    size_mb = arguments.minibatch_size

start_nc = arguments.num_chans

MODEL_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/models/" + phase + expt_name + "/"

if not os.path.exists(MODEL_SAVE_PATH):
    MODEL_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/models/" + phase + expt_name + "/"

IMAGE_SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/" + phase + expt_name + '/'
# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/saved_images/" + phase + expt_name + "/"
# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/real_test_imgs/Out/" + expt_name + "/"

if not os.path.exists(IMAGE_SAVE_PATH):
    # IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/saved_images/" + phase + expt_name + "/"
    # IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/real_test_imgs/Out/" + expt_name + "/"
    pass


vol_dims = [size_mb, image_res, image_res, 12, 1]

os.chdir(FILE_PATH)
hi_list = os.listdir('Hi/')
lo_list = os.listdir('Lo/')
hi_list.sort()
lo_list.sort()

assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"
N = len(hi_list)

indices = list(range(0, N))

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

SRNet = UNet(ph_lo, start_nc)
pred_images = SRNet.output

loss = lossL2(ph_hi, pred_images)
test_loss = 0

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATH + expt_name)

    for iter in range(0, N, size_mb):
        hi_mb, lo_mb = imgLoader(hi_list, lo_list, indices[iter:iter+size_mb])
        test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
        output = sess.run(pred_images, feed_dict=test_feed)

        # print("Test loss: {}".format(sess.run(loss / size_mb, feed_dict=test_feed)))
        test_loss = test_loss + sess.run(loss, feed_dict=test_feed)

        for idx in range(0, size_mb):
            np.save(IMAGE_SAVE_PATH + hi_list[iter+idx][-26:-5] + 'O', output[idx, :, :, :, 0])

    print("Overall loss per image: {}".format(test_loss / N))
