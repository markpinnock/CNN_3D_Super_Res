from argparse import ArgumentParser
import numpy as np
import random
import os
import sys
import tensorflow as tf
import time

sys.path.append('..')
sys.path.append('/home/mpinnock/CNN_3D_Super_Res/scripts/')

from utils.UNet import UNet
from utils.UNet2 import UNet2
from utils.functions import imgLoader, imgLoader2
from utils.losses import lossL2, calcPSNR, calcSSIM
from utils.losses import regFFT, reg3DFFT, regLaplace


parser = ArgumentParser()
parser.add_argument('--file_path', '-fp', help="Data file path", type=str)
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--resolution', '-r', help="Resolution e.g. 512, 128", type=int, nargs='?', const=512, default=512)
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=8, default=8)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int)
parser.add_argument('--folds', '-f', help="Number of cross-validation folds", type=int, nargs='?', const=0, default=0)
parser.add_argument('--crossval', '-c', help="Fold number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--gpu', '-g', help="GPU number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--lamb', '-l', help="Regularisation hyperparameter", type=float, nargs='?', const=0.0, default=0.0)
arguments = parser.parse_args()

if arguments.file_path == None:
    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

    if not os.path.exists(FILE_PATH):
        FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

else:
    FILE_STEM = arguments.file_path
    FILE_PATH = FILE_STEM + 'data/'

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

if fold >= num_folds and num_folds != 0:
   raise ValueError("Fold number cannot be greater or equal to number of folds")

gpu_number = arguments.gpu
LAMBDA = arguments.lamb

MODEL_SAVE_PATH = "/home/mpinnock/models/" + expt_name + "/"

if not os.path.exists(MODEL_SAVE_PATH) and num_folds == 0:
    os.mkdir(MODEL_SAVE_PATH)

if arguments.file_path == None:
    LOG_SAVE_NAME = "C:/Users/rmappin/" + expt_name

    if not os.path.exists("C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_logs/"):
        LOG_SAVE_NAME = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_logs/" + expt_name
else:
    LOG_SAVE_NAME = FILE_STEM + "reports/" + expt_name

ETA = 0.001
hi_vol_dims = [size_mb, image_res, image_res, 12, 1]
lo_vol_dims = [size_mb, image_res, image_res, 12, 1]

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

N = len(hi_list)
indices = list(range(0, N))

if num_folds == 0:
    hi_train = hi_list
    lo_train = lo_list
    N_train = len(hi_train)
    train_indices = indices

else:
    num_in_fold = int(N / num_folds)
    val_indices = indices[fold*num_in_fold:(fold+1)*num_in_fold]
    train_indices = list(set(indices) - set(val_indices))
    N_train = len(train_indices)
    N_val = len(val_indices)

with tf.device('/device:GPU:{}'.format(gpu_number)):
    ph_hi = tf.placeholder(tf.float32, hi_vol_dims)
    ph_lo = tf.placeholder(tf.float32, lo_vol_dims)
    SRNet = UNet2(ph_lo, start_nc)
    pred_images = SRNet.output
    L2 = lossL2(ph_hi, pred_images)
    # reg_term = regLaplace(ph_hi, pred_images)
    reg_term = reg3DFFT(ph_hi, pred_images)
    total_loss = L2 + (LAMBDA * reg_term)
    train_op = tf.train.AdamOptimizer(learning_rate=ETA).minimize(total_loss)

log_file = open(LOG_SAVE_NAME, 'w')
log_file.write("nc{}_ep{}_n{}_fft{}".format(start_nc, num_epoch, N, LAMBDA))
print("nc{}_ep{}_n{}_fft{}".format(start_nc, num_epoch, N, LAMBDA))

if num_folds != 0:
    log_file.write("_cv{}\n".format(fold))
    print("_cv{}".format(fold))
else:
    log_file.write("\n")

start_time = time.time()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.global_variables_initializer())

    for ep in range(num_epoch):
        random.shuffle(train_indices)
        train_L2 = 0
        train_reg = 0
        val_L2 = 0
        val_pSNR = 0
        val_SSIM = 0

        for iter in range(0, N_train, size_mb):
            if any(idx >= N_train for idx in list(range(iter, iter+size_mb))):
                continue
            
            else:
                hi_mb, lo_mb = imgLoader(hi_list, lo_list, train_indices[iter:iter+size_mb])
                train_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
                sess.run(train_op, feed_dict=train_feed)
                train_L2 = train_L2 + sess.run(L2, feed_dict=train_feed)

                if LAMBDA != 0:
                    train_reg = train_reg + sess.run(reg_term, feed_dict=train_feed)

        print('Epoch {} train loss: {}'.format(ep, train_L2 / (N_train - (N_train % size_mb))))
        log_file.write('Epoch {} train loss: {}'.format(ep, train_L2 / (N_train - (N_train % size_mb))))
        
        if LAMBDA != 0:
            print('Reg loss: {}'.format(train_reg / (N_train - (N_train % size_mb))))
            log_file.write(', reg loss: {}'.format(train_reg / (N_train - (N_train % size_mb))))
            print('Total loss {}'.format((np.float(train_L2) + (LAMBDA * train_reg)) / (N_train - (N_train % size_mb))))
            log_file.write(', total loss {}\n'.format((np.float(train_L2) + (LAMBDA * train_reg)) / (N_train - (N_train % size_mb))))
                
        else:
            log_file.write('\n')
    
        if num_folds != 0:
            for iter in range(0, N_val, size_mb):
                if any(idx >= N_val for idx in list(range(iter, iter+size_mb))):
                    continue

                else:
                    hi_mb, lo_mb = imgLoader(hi_list, lo_list, val_indices[iter:iter+size_mb])
                    val_feed = {ph_hi: hi_mb, ph_lo: lo_mb}
                    val_L2 += sess.run(L2, feed_dict=val_feed)
                    val_pSNR += calcPSNR(sess.run(ph_hi, feed_dict=val_feed), sess.run(pred_images, feed_dict=val_feed))
                    val_SSIM += calcSSIM(sess.run(ph_hi, feed_dict=val_feed), sess.run(pred_images, feed_dict=val_feed))

            print('Epoch {} val loss: {}'.format(ep, val_L2 / (N_val - (N_val % size_mb))))
            log_file.write('Epoch {} val loss: {}\n'.format(ep, val_L2 / (N_val - (N_val % size_mb))))

    if num_folds == 0:
        # pass
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, expt_name))
    
    else:
        print('N_val = {}'.format(N_val - (N_val % size_mb)))
        print('Sum val loss for fold {}: {}'.format(fold, val_L2))
        print('pSNR per vol: {}'.format(val_pSNR/ N_val))
        print('SSIM per vol: {}'.format(val_SSIM / N_val))
        log_file.write('N_val = {}\n'.format(N_val - (N_val % size_mb)))
        log_file.write('Summed validation loss for fold {}: {}\n'.format(fold, val_L2))
        log_file.write('pSNR per vol: {}\n'.format(val_pSNR/ N_val))
        log_file.write('SSIM per vol: {}\n'.format(val_SSIM / N_val))
        
    print("Total time: {:.2f} min".format((time.time() - start_time) / 60))
    log_file.write("Total time: {:.2f} min\n".format((time.time() - start_time) / 60))
    print("Parameters: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

log_file.close()
