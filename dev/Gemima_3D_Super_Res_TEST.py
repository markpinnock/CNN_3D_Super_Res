import numpy as np
import random
import os
import sys
import tensorflow as tf
import time

sys.path.append('..')

from utils.UNet import UNet
from utils.UNet2 import UNet2
from utils.functions import imgLoader2
from utils.losses import lossL2, regFFT, regLaplace


# hi_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"

MODEL_NAME = 'test4fullres'

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"
MODEL_SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_res/models/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"
    MODEL_SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_res/models/"

num_epoch = 50
size_mb = 8
num_test = 8
eta = 0.001
hi_vol_dims = [size_mb, 128, 128, 12, 1]
lo_vol_dims = [size_mb, 128, 128, 3, 1]
# np.set_printoptions(precision=2)
random.seed(10)

os.chdir(FILE_PATH)
hi_list = os.listdir('Hi/')
lo_list = os.listdir('Lo/')
hi_list.sort()
lo_list.sort()
temp_list = list(zip(hi_list, lo_list))
random.shuffle(temp_list)
hi_list, lo_list = zip(*temp_list)
assert len(hi_list) == len(lo_list), "Unequal numbers of high and low res"

test_hi_list = hi_list[0:num_test]
test_lo_list = lo_list[0:num_test]
train_hi_list = hi_list[num_test:]
train_lo_list = lo_list[num_test:]

N = len(train_hi_list)
indices = list(range(0, N))

ph_hi = tf.placeholder(tf.float32, hi_vol_dims)
ph_lo = tf.placeholder(tf.float32, lo_vol_dims)

SRNet = UNet2(ph_lo, 4)
pred_images = SRNet.output
L2 = lossL2(ph_hi, pred_images)
reg = regFFT(ph_hi, pred_images)
loss = L2 + 0 * reg
train_op = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

start_time = time.time()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for ep in range(num_epoch):
        random.shuffle(indices)

        for iter in range(0, N - size_mb, size_mb):
            hi_mb, lo_mb = imgLoader2(train_hi_list, train_lo_list, indices[iter:iter+size_mb])
            train_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
            sess.run(train_op, feed_dict=train_feed)

        print('Epoch: {}, Training loss: {}, {}'.format(ep, sess.run(L2, feed_dict=train_feed), sess.run(reg, feed_dict=train_feed)))

    print('SAVED')

    hi_mb, lo_mb = imgLoader2(test_hi_list, test_lo_list, list(range(0, size_mb)))
    test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
    print('Test loss: {}'.format(sess.run(L2, feed_dict=test_feed)))
    
print("Total time: {}".format(time.time() - start_time))

