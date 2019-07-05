import numpy as np
import os
import pytest
import sys

sys.path.append('..')

from utils.functions import lossL2


def imgGen():
    img1 = np.zeros(64, 64)
    img2 = np.ones(64, 64)

    img1[16:47, 16:47] = 1
    img2 = img2 - img1

    ph_1 = tf.placeholder(tf.float32, vol_dims)