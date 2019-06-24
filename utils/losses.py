import tensorflow as tf

def lossL1(batch_A, batch_B):
    return tf.reduce_sum(tf.abs(batch_A - batch_B), axis=[0, 1, 2, 3])


def lossL2(batch_A, batch_B):
    return tf.reduce_sum(tf.square(batch_A - batch_B), axis=[0, 1, 2, 3])