import math
import numpy as np
import tensorflow as tf


# Defines the convolution filter for feature extraction (1 feature per channel)
def convKernel(channels, name='W'):
    with tf.variable_scope(name):
        shape_kernel = [3, 3, 3] + channels
        return tf.get_variable(name, shape=shape_kernel, initializer=tf.contrib.layers.xavier_initializer())


# 3D convolutional layer that takes data input and convolves with filter
def convLayer(layer_input, channels, name='conv_layer'):
    with tf.variable_scope(name):
        strides = [1, 1, 1, 1, 1]
        w = convKernel(channels)
        return tf.nn.relu(tf.nn.conv3d(layer_input, w, strides, padding='SAME'))


# Downsamples data by 2 in all spatial dimensions
def maxPoolLayer(layer_input, name='max_pool_layer'):
    with tf.variable_scope(name):
        strides = [1, 2, 2, 2, 1]
        pool_kernel = [1, 2, 2, 2, 1]
        return tf.nn.max_pool3d(layer_input, pool_kernel, strides, padding='SAME')

        
# Transpose convolutional layer that also upsamples by 2
def convTransLayer(layer_input, skip_input, channels, name='conv_trans_layer'):
    with tf.variable_scope(name):
        strides = [1, 2, 2, 2, 1]
        w = convKernel(channels)
        skip_shape = skip_input.get_shape().as_list()
        return tf.nn.conv3d_transpose(layer_input, w, output_shape=skip_shape, strides=strides, padding='SAME')


# Allows data to skip from encoder side to decode side
def convSkipLayer(layer_input, skip_input, channels, name='conv_skip_layer'):
    with tf.variable_scope(name):
        strides = [1, 1, 1, 1, 1]
        w = convKernel(channels)
        return tf.nn.relu(tf.nn.conv3d(layer_input + skip_input, w, strides, padding='SAME'))


class UNet:

    def __init__(self, image_batch, start_nc):
        # Number of feature maps in each layer's convolutional filters
        base2_pow_nc = int(math.log2(start_nc))
        self.nc = [2**n for n in range(base2_pow_nc, base2_pow_nc + 5)]
        # npgain = np.ones([1, 32, 32, 1, 64], dtype=np.float32)
        # npgain[0, :, :, :, 63] = -100000
        # gain = tf.convert_to_tensor(npgain, dtype=tf.float32)
      
        # Encoding portion: alternating feature extraction and downsampling
        dn_pre_layer_0 = convLayer(name='dn_conv_layer_0', layer_input=image_batch, channels=[1, self.nc[0]])
        dn_layer_0 = maxPoolLayer(name='dn_max_pool_layer_0', layer_input=dn_pre_layer_0)
        dn_pre_layer_1 = convLayer(name='dn_conv_layer_1', layer_input=dn_layer_0, channels=[self.nc[0], self.nc[1]])
        dn_layer_1 = maxPoolLayer(name='dn_max_pool_layer_1', layer_input=dn_pre_layer_1)
        dn_pre_layer_2 = convLayer(name='dn_conv_layer_2', layer_input=dn_layer_1, channels=[self.nc[1], self.nc[2]])
        dn_layer_2 = maxPoolLayer(name='dn_max_pool_layer_2', layer_input=dn_pre_layer_2)
        dn_pre_layer_3 = convLayer(name='dn_conv_layer_3', layer_input=dn_layer_2, channels=[self.nc[2], self.nc[3]])
        dn_layer_3 = maxPoolLayer(name='dn_max_pool_layer_3', layer_input=dn_pre_layer_3)
        # mod_layer=tf.math.multiply(dn_layer_3, gain)
        mod_layer = dn_layer_3
        layer_4 = convLayer(name='conv_layer_4', layer_input=mod_layer, channels=[self.nc[3], self.nc[4]])

        # Decoding portion: upsampling, combining with residual data from encoding portion
        up_pre_layer_3 = convTransLayer(name='up_conv_layer_3', layer_input=layer_4, skip_input=dn_pre_layer_3, channels=[self.nc[3], self.nc[4]])
        up_layer_3 = convSkipLayer(name='up_skip_layer_3', layer_input=up_pre_layer_3, skip_input=dn_pre_layer_3, channels=[self.nc[3], self.nc[3]])
        up_pre_layer_2 = convTransLayer(name='up_conv_layer_2', layer_input=up_layer_3, skip_input=dn_pre_layer_2, channels=[self.nc[2], self.nc[3]])
        up_layer_2 = convSkipLayer(name='up_skip_layer_2', layer_input=up_pre_layer_2, skip_input=dn_pre_layer_2, channels=[self.nc[2], self.nc[2]])
        up_pre_layer_1 = convTransLayer(name='up_conv_layer_1', layer_input=up_layer_2, skip_input=dn_pre_layer_1, channels=[self.nc[1], self.nc[2]])
        up_layer_1 = convSkipLayer(name='up_skip_layer_1', layer_input=up_pre_layer_1, skip_input=dn_pre_layer_1, channels=[self.nc[1], self.nc[1]])
        up_pre_layer_0 = convTransLayer(name='up_conv_layer_0', layer_input=up_layer_1, skip_input=dn_pre_layer_0, channels=[self.nc[0], self.nc[1]])
        up_layer_0 = convSkipLayer(name='up_skip_layer_0', layer_input=up_pre_layer_0, skip_input=dn_pre_layer_0, channels=[self.nc[0], self.nc[0]])

        up_layer_f = convLayer(name='up_conv_layer_f', layer_input=up_layer_0, channels=[self.nc[0], 1])
        
        self.dnlayer_0 = dn_pre_layer_0
        self.dnlayer_1 = dn_pre_layer_1
        self.dnlayer_2 = dn_pre_layer_2
        self.dnlayer_3 = dn_pre_layer_3
        self.layer_4 = layer_4
        self.uplayer_3 = up_pre_layer_3
        self.uplayer_2 = up_pre_layer_2
        self.uplayer_1 = up_pre_layer_1
        self.uplayer_0 = up_pre_layer_0
        self.uplayer_f = up_layer_f

        # Output
        self.output = up_layer_f
