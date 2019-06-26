import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


file_path = "G:/PhD/Seg_Data/nrrd/"
file_list = os.listdir(file_path)

vol = nrrd.read(file_path + file_list[0])
seg = nrrd.read(file_path + file_list[1])
vol = vol[0]
seg = seg[0]
seg = seg[0:4, :, :, :]

for i in range(0, 4):
    for j in range(0, seg.shape[3]):
        if np.sum(seg[i, :, :, j]) == 0:
            continue
        
        if i != 0 and j % 10 != 0 and seg.shape[3] > 3:
            continue

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.fliplr((vol[:, :, j] * seg[i, :, :, j]).T), cmap='gray', origin='lower')
        ax[1].imshow(np.fliplr(vol[:, :, j].T), cmap='gray', origin='lower')
        plt.show()

np.save(file_path + file_list[0][:-5], vol)
np.save(file_path + file_list[1][:-9], seg)
