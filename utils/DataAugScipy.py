import numpy as np
import scipy.ndimage as sci
import sys


class DataAugmentationScipy:

    def __init__(self, img_dims):
        if len(img_dims) != 5:
            print(img_dims, file=sys.stderr)
            raise ValueError('Must be 5 image volume dimensions')
        else:
            self._img_dims = img_dims

        self._ident_mat = np.identity(2)

    def flipMat(self, prob):
        # t1 = np.random.binomial(1, prob)
        t2 = np.random.binomial(1, prob)

        flip_mat = np.copy(self._ident_mat)

        # if t1:
        #     flip_mat[0, 0] = -1

        if t2:
            flip_mat[1, 1] = -1
        
        return flip_mat
    
    def rotMat(self, theta):
        for _ in range(10):
            theta = np.random.normal(0, theta)

            if abs(theta) < 90:
                break
        
        theta = theta / 180 * np.pi
    
        rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
         )

        return rot_mat
    
    def scaleMat(self, scale):
        # z = np.random.normal(1, scale)
        z = np.random.uniform(1 - scale, 1 + scale)
        scale_mat = np.copy(self._ident_mat)
        scale_mat[0, 0] = z
        scale_mat[1, 1] = z

        return scale_mat
    
    def shearMat(self, phi):
        phi = phi / 180 * np.pi
        phi = np.random.uniform(-phi, phi)

        p = False
        p = bool(np.random.binomial(1, 0.5))

        shear_mat = np.copy(self._ident_mat)

        if p:
            shear_mat[0, 1] = phi
        else:
            shear_mat[1, 0] = phi

        return shear_mat
    
    def transMatGen(self, flip, rot, scale, shear):
        trans_mat = self._ident_mat

        if flip == None:
            pass
        elif flip < 0 or flip > 1:
            raise ValueError("Flip probability out must be between 0 and 1") 
        else:
            trans_mat = np.matmul(trans_mat, self.flipMat(flip))
        
        if rot != None:
            trans_mat = np.matmul(trans_mat, self.rotMat(rot))
        
        if scale != None:
            trans_mat = np.matmul(trans_mat, self.scaleMat(scale))
        
        if shear != None:
            trans_mat = np.matmul(trans_mat, self.shearMat(shear))
        
        return trans_mat

    def warpImg(self, hi_vol, lo_vol, flip=None, rot=None, scale=None, shear=None):
        h_off = self._img_dims[1] / 2
        w_off = self._img_dims[2] / 2
        centre = np.array([[h_off], [w_off]])

        for vol in range(self._img_dims[0]):
            trans_mat = self.transMatGen(flip, rot, scale, shear)
            offset = centre - trans_mat.dot(centre)

            for img in range(self._img_dims[3]):
                hi_vol[vol, :, :, img, 0] = sci.interpolation.affine_transform(
                    hi_vol[vol, :, :, img, 0],
                    trans_mat,
                    offset=(int(offset[0]), int(offset[1])),
                    mode='constant',
                    cval=0.0)
                
                lo_vol[vol, :, :, img, 0] = sci.interpolation.affine_transform(
                    lo_vol[vol, :, :, img, 0],
                    trans_mat,
                    offset=(int(offset[0]), int(offset[1])),
                    mode='constant',
                    cval=0.0)
        
        return hi_vol, lo_vol