import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sci
import sys

sys.path.append('..')

from utils.functions import imgLoader

FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/NPY_Vols/"

os.chdir(FILE_PATH)
hi_list = os.listdir('Hi/')
lo_list = os.listdir('Lo/')
hi_list.sort()
lo_list.sort()

hi_vol, lo_vol = imgLoader(hi_list, lo_list, [0, 1])


class DataAugmentation:

    def __init__(self, img_dims, eta):
        if len(img_dims) != 5:
            print(img_dims, file=sys.stderr)
            raise ValueError('Must be 5 image volume dimensions')
        else:
            self._img_dims = img_dims

        if eta < 0:
            print(eta, file=sys.stderr)
            raise ValueError('Eta must be greater than or equal to zero')
        else:
            self._eta = eta

        self._ident_mat = np.identity(4)
        self._flat_coords = self._coordGen()
        self._aff_mat = np.copy(self._ident_mat)
    
    def _coordGen(self):
        y_grid, x_grid, z_grid = np.meshgrid(
            np.linspace(0, self._img_dims[1] - 1, self._img_dims[1]),
            np.linspace(0, self._img_dims[2] - 1, self._img_dims[2]),
            np.linspace(0, self._img_dims[3] - 1, self._img_dims[3]))

        return np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), np.ones(x_grid.shape[0] * x_grid.shape[1] * x_grid.shape[2])])
    
    def _defFieldGen(self):
        new_coords = np.matmul(self._aff_mat, self._flat_coords)
        x_grid_new = np.reshape(new_coords[0, :], self._img_dims[1:4])
        y_grid_new = np.reshape(new_coords[1, :], self._img_dims[1:4])
        z_grid_new = np.reshape(new_coords[2, :], self._img_dims[1:4])
        return np.concatenate((x_grid_new[:, :, :, np.newaxis], y_grid_new[:, :, :, np.newaxis], z_grid_new[:, :, :, np.newaxis]), axis=3)

    def flipMat(self):
        t1 = np.random.uniform(-1, 1)
        t2 = np.random.uniform(-1, 1)

        flip_mat = np.copy(self._ident_mat)

        if self._eta == 0:
            pass
        else:
            if t1 < 0:
                flip_mat[0, 0] = -1

            if t2 < 0:
                flip_mat[1, 1] = -1
        
        self.transMat(self._img_dims[1] / 2, self._img_dims[2] / 2)
        self._aff_mat = np.matmul(self._aff_mat, flip_mat)
        self.transMat(-self._img_dims[1] / 2, -self._img_dims[2] / 2)
        return self
    
    def rotMat(self):
        theta = np.random.uniform(-0.8, 0.8)
        theta *= self._eta
    
        rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0],
         [np.sin(theta), np.cos(theta), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
         )

        self.transMat(self._img_dims[1] / 2, self._img_dims[2] / 2)
        self._aff_mat = np.matmul(self._aff_mat, rot_mat)
        self.transMat(-self._img_dims[1] / 2, -self._img_dims[2] / 2)
        return self
    
    def scaleMat(self):
        if self._eta >= 1:
            z = np.random.uniform(0.5 / self._eta, 1.2)
        elif self._eta == 0:
            z = 1
        else:
            z = np.random.uniform(0.8, 1.2)

        scale_mat = np.copy(self._ident_mat)
        scale_mat[0, 0] = z
        scale_mat[1, 1] = z

        self.transMat(self._img_dims[1] / 2, self._img_dims[2] / 2)
        self._aff_mat = np.matmul(self._aff_mat, scale_mat)
        self.transMat(-self._img_dims[1] / 2, -self._img_dims[2] / 2)
        return self
    
    def shearMat(self):
        s_x = np.random.uniform(-0.2, 0.2)
        s_y = np.random.uniform(-0.2, 0.2)
        s_x *= self._eta
        s_y *= self._eta

        if (s_x > 0 and s_y < 0) or (s_x < 0 and s_y > 0):
            s_y = -s_y

        shear_mat = np.copy(self._ident_mat)
        shear_mat[0, 1] = s_x
        shear_mat[1, 0] = s_y

        self.transMat(self._img_dims[1] / 2, self._img_dims[2] / 2)
        self._aff_mat = np.matmul(self._aff_mat, shear_mat)
        self.transMat(-self._img_dims[1] / 2, -self._img_dims[2] / 2)
        return self
    
    def transMat(self, t_x=None, t_y=None):
        if t_x == None or t_y == None:
            t_x = np.random.uniform(-self._img_dims[1] / 8, self._img_dims[1] / 8)
            t_y = np.random.uniform(-self._img_dims[2] / 8, self._img_dims[2] / 8)
            t_x *= self._eta
            t_y *= self._eta

        trans_mat = np.copy(self._ident_mat)
        trans_mat[0, 3] = t_x
        trans_mat[1, 3] = t_y
        self._aff_mat = np.matmul(self._aff_mat, trans_mat)
        
        return self

    def warpImg(self, hi_vol, lo_vol):
        new_coords = np.stack([self._defFieldGen() for idx in range(self._img_dims[0])], axis=0)
        new_hi_vol = np.zeros(self._img_dims)
        new_lo_vol = np.zeros(self._img_dims)

        for idx in range(self._img_dims[0]):
            new_hi_vol[idx, :, :, :, 0] = sci.interpn(
                (np.arange(self._img_dims[1]), np.arange(self._img_dims[2]), np.arange(self._img_dims[3])),
                hi_vol[idx, :, :, :, 0],
                new_coords[idx, ...],
                method='linear',
                fill_value=0,
                bounds_error=False)
            
            new_lo_vol[idx, :, :, :, 0] = sci.interpn(
                (np.arange(self._img_dims[1]), np.arange(self._img_dims[2]), np.arange(self._img_dims[3])),
                lo_vol[idx, :, :, :, 0],
                new_coords[idx, ...],
                method='linear',
                fill_value=0,
                bounds_error=False)
        
        self._aff_mat = self._ident_mat
        return new_hi_vol, new_lo_vol


DataAug = DataAugmentation([2, 128, 128, 12, 1], 0.5)
new_hi_vol, new_lo_vol = DataAug.shearMat().rotMat().scaleMat().flipMat().warpImg(hi_vol, lo_vol)

for i in range(1):
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(hi_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 1].imshow(new_hi_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[0, 2].imshow(new_hi_vol[1, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 0].imshow(lo_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 1].imshow(new_lo_vol[0, :, :, i, 0].T, origin='lower', cmap='gray')
    axs[1, 2].imshow(new_lo_vol[1, :, :, i, 0].T, origin='lower', cmap='gray')
    plt.show()