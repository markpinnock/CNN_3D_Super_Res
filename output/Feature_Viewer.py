import matplotlib.pyplot as plt
import numpy as np
import os


FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/features/nc8_ep20_n1026_features/layers/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/features/nc8_ep20_n1026_features/layers/"

dnlayer0 = np.load(FILE_PATH + 'dnlayer0.npy')
dnlayer1 = np.load(FILE_PATH + 'dnlayer1.npy')
dnlayer2 = np.load(FILE_PATH + 'dnlayer2.npy')
dnlayer3 = np.load(FILE_PATH + 'dnlayer3.npy')
layer4 = np.load(FILE_PATH + 'layer4.npy')
uplayer3 = np.load(FILE_PATH + 'uplayer3.npy')
uplayer2 = np.load(FILE_PATH + 'uplayer2.npy')
uplayer1 = np.load(FILE_PATH + 'uplayer1.npy')
uplayer0 = np.load(FILE_PATH + 'uplayer0.npy')

for i in range(1):
    fig, axs = plt.subplots(3, 3)

    axs[0, 0].imshow(dnlayer0[0, :, :, 10, 0].T, cmap='gray', origin='lower')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(dnlayer1[0, :, :, 5, 15].T, cmap='gray', origin='lower')
    axs[0, 1].axis('off')
    axs[0, 2].imshow(dnlayer2[0, :, :, 2, 2].T, cmap='gray', origin='lower') # 2, 9, 21
    axs[0, 2].axis('off')
    axs[1, 0].imshow(dnlayer3[0, :, :, 1, 63].T, cmap='gray', origin='lower') # 12, 15, 27, 49, 50, 63
    axs[1, 0].axis('off')
    axs[1, 1].imshow(layer4[0, :, :, 0, 48].T, cmap='gray', origin='lower') # 16, 48, 113
    axs[1, 1].axis('off')
    axs[1, 2].imshow(uplayer3[0, :, :, 1, 14].T, cmap='gray', origin='lower') # 14
    axs[1, 2].axis('off')
    axs[2, 0].imshow(uplayer2[0, :, :, 2, 31].T, cmap='gray', origin='lower')
    axs[2, 0].axis('off')
    axs[2, 1].imshow(uplayer1[0, :, :, 5, 15].T, cmap='gray', origin='lower')
    axs[2, 1].axis('off')
    axs[2, 2].imshow(uplayer0[0, :, :, 10, 2].T, cmap='gray', origin='lower')
    axs[2, 2].axis('off')
    print(i)
    plt.show()