import math
import numpy as np
import os
import tensorflow as tf


def imgLoader(hi_list, lo_list, indices):
    TENSOR_DIMS = 5

    try:
        hi_mb = [hi_list[i] for i in indices]
        lo_mb = [lo_list[i] for i in indices]
    except:
        raise IndexError("Index out of range")

    if len(hi_mb) != len(lo_mb):
        raise ValueError("hi_mb and lo_mb lengths do not match")

    if [vol[-26:-5] for vol in hi_mb] != [vol[-26:-5] for vol in lo_mb]:
        raise ValueError("hi_mb and lo_mb names do not match")

    if 'L' in [vol[-26:-4] for vol in hi_mb]:
        raise ValueError("Lo vol in hi_mb")

    if 'H' in [vol[-26:-4] for vol in lo_mb]:
        raise ValueError("Hi vol in lo_mb")  

    hi_img = np.expand_dims(np.stack([np.float32(np.load(img)) for img in hi_mb], axis=0), axis=4)
    lo_img = np.expand_dims(np.stack([np.float32(np.load(img)) for img in lo_mb], axis=0), axis=4)

    if hi_img.shape != lo_img.shape:
        raise ValueError("hi_mb and lo_mb tensor shapes do not match")

    if len(hi_img.shape) != TENSOR_DIMS:
        raise ValueError("Tensor dimensions not correct")

    return hi_img, lo_img


def lossL1(batch_A, batch_B):
    return tf.reduce_sum(tf.abs(batch_A - batch_B), axis=[0, 1, 2, 3])


def lossL2(batch_A, batch_B):
    return tf.reduce_sum(tf.square(batch_A - batch_B), axis=[0, 1, 2, 3])


def convKernel(channels, name='W'):
    with tf.variable_scope(name):
        shape_kernel = [3, 3, 3] + channels
        return tf.get_variable(name, shape=shape_kernel, initializer=tf.contrib.layers.xavier_initializer())


def convLayer(layer_input, channels, name='conv_layer'):
    with tf.variable_scope(name):
        strides = [1, 1, 1, 1, 1]
        w = convKernel(channels)
        conv = tf.nn.conv3d(layer_input, w, strides, padding='SAME')
        norm = tf.contrib.layers.batch_norm(conv)
        return tf.nn.relu(norm)


def maxPoolLayer(layer_input, name='max_pool_layer'):
    with tf.variable_scope(name):
        strides = [1, 2, 2, 2, 1]
        pool_kernel = [1, 2, 2, 2, 1]
        return tf.nn.max_pool3d(layer_input, pool_kernel, strides, padding='SAME')


def convTransLayer(layer_input, skip_input, channels, name='conv_trans_layer'):
    with tf.variable_scope(name):
        strides = [1, 2, 2, 2, 1]
        w = convKernel(channels)
        skip_shape = skip_input.get_shape().as_list()
        return tf.nn.conv3d_transpose(layer_input, w, output_shape=skip_shape, strides=strides, padding='SAME')


def convSkipLayer(layer_input, skip_input, channels, name='conv_skip_layer'):
    with tf.variable_scope(name):
        strides = [1, 1, 1, 1, 1]
        w = convKernel(channels)
        return tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(layer_input + skip_input, w, strides, padding='SAME')))


class UNet:

    def __init__(self, image_batch, start_nc):
        base2_pow_nc = int(math.log2(start_nc))
        self.nc = [2**n for n in range(base2_pow_nc, base2_pow_nc + 5)]
      
        dn_pre_layer_0 = convLayer(name='dn_conv_layer_0', layer_input=image_batch, channels=[1, self.nc[0]])
        dn_layer_0 = maxPoolLayer(name='dn_max_pool_layer_0', layer_input=dn_pre_layer_0)
        dn_pre_layer_1 = convLayer(name='dn_conv_layer_1', layer_input=dn_layer_0, channels=[self.nc[0], self.nc[1]])
        dn_layer_1 = maxPoolLayer(name='dn_max_pool_layer_1', layer_input=dn_pre_layer_1)
        dn_pre_layer_2 = convLayer(name='dn_conv_layer_2', layer_input=dn_layer_1, channels=[self.nc[1], self.nc[2]])
        dn_layer_2 = maxPoolLayer(name='dn_max_pool_layer_2', layer_input=dn_pre_layer_2)
        dn_pre_layer_3 = convLayer(name='dn_conv_layer_3', layer_input=dn_layer_2, channels=[self.nc[2], self.nc[3]])
        dn_layer_3 = maxPoolLayer(name='dn_max_pool_layer_3', layer_input=dn_pre_layer_3)
        
        layer_4 = convLayer(name='conv_layer_4', layer_input=dn_layer_3, channels=[self.nc[3], self.nc[4]])
        
        up_pre_layer_3 = convTransLayer(name='up_conv_layer_3', layer_input=layer_4, skip_input=dn_pre_layer_3, channels=[self.nc[3], self.nc[4]])
        up_layer_3 = convSkipLayer(name='up_skip_layer_3', layer_input=up_pre_layer_3, skip_input=dn_pre_layer_3, channels=[self.nc[3], self.nc[3]])
        up_pre_layer_2 = convTransLayer(name='up_conv_layer_2', layer_input=up_layer_3, skip_input=dn_pre_layer_2, channels=[self.nc[2], self.nc[3]])
        up_layer_2 = convSkipLayer(name='up_skip_layer_2', layer_input=up_pre_layer_2, skip_input=dn_pre_layer_2, channels=[self.nc[2], self.nc[2]])
        up_pre_layer_1 = convTransLayer(name='up_conv_layer_1', layer_input=up_layer_2, skip_input=dn_pre_layer_1, channels=[self.nc[1], self.nc[2]])
        up_layer_1 = convSkipLayer(name='up_skip_layer_1', layer_input=up_pre_layer_1, skip_input=dn_pre_layer_1, channels=[self.nc[1], self.nc[1]])
        up_pre_layer_0 = convTransLayer(name='up_conv_layer_0', layer_input=up_layer_1, skip_input=dn_pre_layer_0, channels=[self.nc[0], self.nc[1]])
        up_layer_0 = convSkipLayer(name='up_skip_layer_0', layer_input=up_pre_layer_0, skip_input=dn_pre_layer_0, channels=[self.nc[0], self.nc[0]])

        up_layer_f = convLayer(name='up_conv_layer_f', layer_input=up_layer_0, channels=[self.nc[0], 1])
        self.downlayer = dn_layer_2
        self.output = tf.nn.relu(up_layer_f)
