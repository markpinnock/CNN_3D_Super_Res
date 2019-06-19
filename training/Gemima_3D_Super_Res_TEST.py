import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import tensorflow as tf

sys.path.append('..')

from utils.Gemima_Utils import UNet, imgLoader, lossL2


# hi_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Hi/"
# lo_path = "G:/PhD/Super_Res_Data/Toshiba_Vols/NPY/Lo/"

model_name = 'test4fullres'
hi_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/Hi/"
lo_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/Lo/"
model_save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_res/models/"

num_epoch = 5
size_mb = 8
num_test = 8
eta = 0.03
vol_dims = [size_mb, 128, 128, 12, 1]
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

test_hi_list = hi_list[0:num_test]
test_lo_list = lo_list[0:num_test]
train_hi_list = hi_list[num_test:]
train_lo_list = lo_list[num_test:]

N = len(train_hi_list)
indices = list(range(0, N))

ph_hi = tf.placeholder(tf.float32, vol_dims)
ph_lo = tf.placeholder(tf.float32, vol_dims)

SRNet = UNet(ph_lo, 2)
pred_images = SRNet.output
loss = lossL2(ph_hi, pred_images)
train_op = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for ep in range(num_epoch):
        random.shuffle(indices)

        for iter in range(0, N - size_mb, size_mb):
            hi_mb, lo_mb = imgLoader(train_hi_list, train_lo_list, indices[iter:iter+size_mb])
            train_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
            sess.run(train_op, feed_dict=train_feed)

        print('Epoch: {}, Training loss: {}'.format(ep, sess.run(loss / size_mb, feed_dict=train_feed)))

    print('SAVED')

    hi_mb, lo_mb = imgLoader(test_hi_list, test_lo_list, list(range(0, size_mb)))
    test_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
    print('Test loss: {}'.format(sess.run(loss / size_mb, feed_dict=test_feed)))
    


