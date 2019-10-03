import keras.preprocessing.image as preproc
import numpy as np
import sys


class DataAugmentationKeras(preproc.ImageDataGenerator):

    def __init__(self, img_dims):
        super(DataAugmentationKeras, self).__init__(data_format='channels_last')

        if len(img_dims) != 5:
            print(img_dims, file=sys.stderr)
            raise ValueError('Must be 5 image volume dimensions')
        else:
            self._img_dims = img_dims
    
    def _paramGen(self):
        theta = np.random.uniform(-45, 45)
        # scale = np.random.uniform(0.9, 1.2)
        vert = bool(np.random.binomial(1, 0.4))

        param_dict = {
            'theta': theta,
            'tx': 0.0,
            'ty': 0.0,
            'shear': 0.0,
            'zx': 0.0,
            'zy': 0.0,
            'flip_horizontal': False,
            'flip_vertical': vert,
            'channel_shift_intensity': None,
            'brightness': None
        }

        return param_dict
    
    def transform(self, hi_mb, lo_mb):
        if len(hi_mb) != len(lo_mb):
            raise ValueError("hi_mb and lo_mb lengths do not match")

        for vol in range(self._img_dims[0]):
            param_dict = self._paramGen()

            for img in range(self._img_dims[3]):
                hi_mb[vol, :, :, img, :] = super().apply_transform(
                    hi_mb[vol, :, :, img, :], param_dict)
                lo_mb[vol, :, :, img, :] = super().apply_transform(
                    lo_mb[vol, :, :, img, :], param_dict)

        return hi_mb, lo_mb