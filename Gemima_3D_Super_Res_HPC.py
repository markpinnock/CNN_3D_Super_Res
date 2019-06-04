from argparse import ArgumentParser
import numpy as np
import random
import os
import tensorflow as tf

from Gemima_Utils import UNet, imgLoader, lossMSE


hi_path = "/home/mpinnock/Hi/"
lo_path = "/home/mpinnock/Lo/"
# hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Hi/" #TEST
# lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Lo/" #TEST
model_save_path = "/home/mpinnock/"

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int)
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

if arguments.epochs == None:
    raise ValueError("Must provide number of epochs")
else:
    num_epoch = arguments.epochs


eta = 0.03
vol_dims = [size_mb, image_res, image_res, 12, 1]
np.set_printoptions(precision=2)
random.seed(10)

hi_list = os.listdir(hi_path)
lo_list = os.listdir(lo_path)
temp_list = list(zip(hi_list, lo_list))
random.shuffle(temp_list)
hi_list, lo_list = zip(*temp_list)
assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"

hi_list = list(map(lambda img: hi_path + img, hi_list))
lo_list = list(map(lambda img: lo_path + img, lo_list))

N = len(hi_list)
indices = list(range(0, N))

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

SRNet = UNet(ph_lo)
pred_images = SRNet.output
loss = lossMSE(ph_hi, pred_images)
train_op = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for ep in range(num_epoch):
        random.shuffle(indices)

        for iter in range(0, N - size_mb, size_mb):
            hi_mb, lo_mb = imgLoader(hi_list, lo_list, indices[iter:iter+size_mb])
            train_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
            sess.run(train_op, feed_dict=train_feed)

        print('Epoch: {}, Training loss: {}'.format(ep, sess.run(loss, feed_dict=train_feed)))

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_save_path, expt_name)) #TEST
