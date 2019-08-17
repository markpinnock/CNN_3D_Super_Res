from argparse import ArgumentParser
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')

from utils.UNet import UNet
from utils.functions import imgLoader
from utils.losses import lossL2


FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/"

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

# IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/nc8_ep20_n1026_features/layers/"
IMAGE_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/Out/"

if not os.path.exists(IMAGE_SAVE_PATH):
    # IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/nc8_ep20_n1026_features/layers/"
    IMAGE_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/features/Out/"

image_res = arguments.resolution

if arguments.minibatch_size == None:
    raise ValueError("Must provide minibatch size")
else:
    size_mb = arguments.minibatch_size

start_nc = arguments.num_chans

MODEL_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/models/" + phase + expt_name + "/"

if not os.path.exists(MODEL_SAVE_PATH):
    MODEL_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_res/models/" + phase + expt_name + "/"

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

        # np.save(IMAGE_SAVE_PATH + 'dnlayer0', sess.run(SRNet.dnlayer_0, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'dnlayer1', sess.run(SRNet.dnlayer_1, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'dnlayer2', sess.run(SRNet.dnlayer_2, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'dnlayer3', sess.run(SRNet.dnlayer_3, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'layer4', sess.run(SRNet.layer_4, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'uplayer3', sess.run(SRNet.uplayer_3, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'uplayer2', sess.run(SRNet.uplayer_2, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'uplayer1', sess.run(SRNet.uplayer_1, feed_dict=test_feed))
        # np.save(IMAGE_SAVE_PATH + 'uplayer0', sess.run(SRNet.uplayer_0, feed_dict=test_feed))


    print("SAVED")
