from argparse import ArgumentParser
import numpy as np
import random
import os
import tensorflow as tf

from Gemima_Utils import UNet, imgLoader, lossL2


HI_PATH = "/home/mpinnock/Data/Hi/"
LO_PATH = "/home/mpinnock/Data/Lo/"
# HI_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Hi/"
# LO_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD Prog/NPY_Vols/Lo/"

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=3, default=3)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int)
parser.add_argument('--folds', '-f', help="Number of cross-validation folds", type=int, nargs='?', const=0, default=0)
parser.add_argument('--crossval', '-c', help="Fold number", type=int, nargs='?', const=0, default=0)
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

start_nc = arguments.num_chans

if arguments.epochs == None:
    raise ValueError("Must provide number of epochs")
else:
    num_epoch = arguments.epochs

num_folds = arguments.folds
fold = arguments.crossval

if fold >= num_folds:
    raise ValueError("Fold number cannot be greater or equal to number of folds")

model_save_path = "/home/mpinnock/Models/" + expt_name + "/"

ETA = 0.03
vol_dims = [size_mb, image_res, image_res, 12, 1]
np.set_printoptions(precision=2)
random.seed(10)

hi_list = os.listdir(HI_PATH)
lo_list = os.listdir(LO_PATH)
temp_list = list(zip(hi_list, lo_list))
random.shuffle(temp_list)
hi_list, lo_list = zip(*temp_list)
assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"

hi_list = list(map(lambda img: HI_PATH + img, hi_list))
lo_list = list(map(lambda img: LO_PATH + img, lo_list))
N = len(hi_list)
indices = list(range(0, N))

if num_folds == 0:
    hi_train = hi_list
    lo_train = lo_list
    N_train = len(hi_train)

else:
    num_in_fold = int(N / num_folds)
    val_indices = indices[fold*num_in_fold:(fold+1)*num_in_fold]
    train_indices = list(set(indices) - set(val_indices))
    N_train = len(train_indices)
    N_val = len(val_indices)

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

SRNet = UNet(ph_lo, start_nc)
print(SRNet.nc) # TEST
pred_images = SRNet.output
loss = lossL2(ph_hi, pred_images)
train_op = tf.train.AdamOptimizer(learning_rate=ETA).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for ep in range(num_epoch):
        random.shuffle(train_indices)
        train_loss = 0

        for iter in range(0, N_train, size_mb):
            if any(idx >= N_train for idx in list(range(iter, iter+size_mb))):
                continue
            
            else:
                hi_mb, lo_mb = imgLoader(hi_list, lo_list, train_indices[iter:iter+size_mb])
                train_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
                sess.run(train_op, feed_dict=train_feed)
                train_loss = train_loss + sess.run(loss, feed_dict=train_feed)

        print('Epoch {} training loss per image: {}'.format(ep, train_loss / (N_train - (N_train % size_mb))))
    
    if num_folds == 0:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(model_save_path, expt_name))
    
    else:
        val_loss = 0

        for iter in range(0, N_val, size_mb):
            if any(idx >= N_val for idx in list(range(iter, iter+size_mb))):
                continue

            else:
                hi_mb, lo_mb = imgLoader(hi_list, lo_list, val_indices[iter:iter+size_mb])
                val_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
                sess.run(train_op, feed_dict=val_feed)
                val_loss = val_loss + sess.run(loss, feed_dict=val_feed)

        print('Summed validation loss for fold {}: {}'.format(fold, val_loss))
        print('Validation loss per image: {}'.format(val_loss / (N_val - (N_val % size_mb))))
