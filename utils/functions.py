import numpy as np
import os
import sys


def diceIndexCalc(vol_A, vol_B):
    numer = 2 * np.sum((vol_A * vol_B), axis=(1, 2, 3))
    denom = np.sum(vol_A, axis=(1, 2, 3)) + np.sum(vol_B, axis=(1, 2, 3)) + 1e-6
    return numer / denom


def imgLoader(hi_list, lo_list, indices):
    TENSOR_DIMS = 5

    try:
        hi_mb = [hi_list[i] for i in indices]
        lo_mb = [lo_list[i] for i in indices]
    except:
        print(indices, file=sys.stderr)
        raise IndexError("Index out of range")

    if len(hi_mb) != len(lo_mb):
        print(hi_mb, lo_mb, file=sys.stderr)
        raise ValueError("hi_mb and lo_mb lengths do not match")

    if [vol[:-5] for vol in hi_mb] != [vol[:-5] for vol in lo_mb]:
        print(hi_mb, lo_mb, file=sys.stderr)
        raise ValueError("hi_mb and lo_mb names do not match")

    if 'L' in [vol[-26:-4] for vol in hi_mb]:
        print(hi_mb, file=sys.stderr)
        raise ValueError("Lo vol in hi_mb")

    if 'H' in [vol[-26:-4] for vol in lo_mb]:
        print(lo_mb, file=sys.stderr)
        raise ValueError("Hi vol in lo_mb")  

    hi_img = np.expand_dims(np.stack([np.float32(np.load('Hi/' + img)) for img in hi_mb], axis=0), axis=4)
    lo_img = np.expand_dims(np.stack([np.float32(np.load('Lo/' + img)) for img in lo_mb], axis=0), axis=4)

    if hi_img.shape != lo_img.shape:
        print(hi_img.shape, lo_img.shape, file=sys.stderr)
        raise ValueError("hi_mb and lo_mb tensor shapes do not match")

    if len(hi_img.shape) != TENSOR_DIMS:
        print(hi_img.shape, lo_img.shape, file=sys.stderr)
        raise ValueError("Tensor dimensions not correct")

    return hi_img, lo_img


