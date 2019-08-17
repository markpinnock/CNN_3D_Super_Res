import matplotlib.pyplot as plt
import numpy as np
import os


FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/features/Out/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/features/Out/"

os.chdir(FILE_PATH)

vol1 = np.load('1.npy')
vol2 = np.load('0.npy')
vol3 = np.load('neg10.npy')

i = 0

fig, axs = plt.subplots(1, 3)
axs[0].imshow(vol1[:, :, i].T, origin='lower', cmap='gray')
axs[1].imshow(vol2[:, :, i].T, origin='lower', cmap='gray')
axs[2].imshow(vol3[:, :, i].T, origin='lower', cmap='gray')
plt.show()
