from argparse import ArgumentParser
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--subject', '-s', help="Subject number", type=str)
parser.add_argument('--plane', '-pl', help="ax, co, sa", type=str)
parser.add_argument('--diff', '-d', help="Difference images", type=str, nargs='?', const='n', default='n')
parser.add_argument('--intervals', '-i', help="ROI x1 x2 y1 y2", type=str)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_string = arguments.expt_name
    expt_names = expt_string.split(' ')

phase = 'Phase_' + arguments.phase + '/'

FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/"
NETWORK_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/Out/" + phase
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/temp/"
# NETWORK_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/temp/Out/"

if arguments.subject == None:
    raise ValueError("Must provide subject number")
else:
    subject = arguments.subject

if arguments.plane == None:
    raise ValueError("Must provide plane")
else:
    plane = arguments.plane

diff = arguments.diff

if arguments.intervals != None:
    interval_string = arguments.intervals
    interval_list = interval_string.split(' ')

hi_list = [img for img in os.listdir(FILE_PATH + 'Hi/') if subject in img]
lo_list = [img for img in os.listdir(FILE_PATH + 'Lo/') if subject in img]
int_list = [img for img in os.listdir(FILE_PATH + 'Int/') if subject in img]
img_list = hi_list + lo_list + int_list

for vol_name in img_list:
    if '_H.nrrd' in vol_name:
        hi_vol, _ = nrrd.read(FILE_PATH + 'Hi/' + vol_name)
    if '_L.nrrd' in vol_name:
        lo_vol, _ = nrrd.read(FILE_PATH + 'Lo/' + vol_name)
    if '_I.nrrd' in vol_name:
        int_vol, _ = nrrd.read(FILE_PATH + 'Int/' + vol_name)
    
    vol_num = vol_name[16]

output_list = os.listdir(NETWORK_PATH + expt_names[0])
output_list = [img for img in output_list if subject in img and img[16] == vol_num]
network_1, _ = nrrd.read(NETWORK_PATH + expt_names[0] + '/' + output_list[0])
output_list = os.listdir(NETWORK_PATH + expt_names[1])
output_list = [img for img in output_list if subject in img and img[16] == vol_num]
network_2, _ = nrrd.read(NETWORK_PATH + expt_names[1] + '/' + output_list[0])
output_list = os.listdir(NETWORK_PATH + expt_names[2])
output_list = [img for img in output_list if subject in img and img[16] == vol_num]
network_3, _ = nrrd.read(NETWORK_PATH + expt_names[2] + '/' + output_list[0])

if len(expt_names) == 4:
    output_list = os.listdir(NETWORK_PATH + expt_names[3])
    output_list = [img for img in output_list if subject in img and img[16] == vol_num]
    network_4, _ = nrrd.read(NETWORK_PATH + expt_names[3] + '/' + output_list[0])

if len(expt_names) == 5:
    output_list = os.listdir(NETWORK_PATH + expt_names[3])
    output_list = [img for img in output_list if subject in img and img[16] == vol_num]
    network_4, _ = nrrd.read(NETWORK_PATH + expt_names[3] + '/' + output_list[0])
    output_list = os.listdir(NETWORK_PATH + expt_names[4])
    output_list = [img for img in output_list if subject in img and img[16] == vol_num]
    network_5, _ = nrrd.read(NETWORK_PATH + expt_names[4] + '/' + output_list[0])

vol_dims = hi_vol.shape

if diff == 'y':
    lo_vol = hi_vol - lo_vol
    int_vol = hi_vol - int_vol
    network_1 = hi_vol - network_1
    network_2 = hi_vol - network_2
    network_3 = hi_vol - network_3

    if len(expt_names) == 4:
        network_4 = hi_vol - network_4

    if len(expt_names) == 5:
        network_4 = hi_vol - network_4
        network_5 = hi_vol - network_5

# Normalise
# hi_vol = (hi_vol - hi_vol.min()) / (hi_vol.max() - hi_vol.min())
# lo_vol = (lo_vol - lo_vol.min()) / (lo_vol.max() - lo_vol.min())
# int_vol = (int_vol - int_vol.min()) / (int_vol.max() - int_vol.min())
# network_1 = (network_1 - network_1.min()) / (network_1.max() - network_1.min())
# network_2 = (network_2 - network_2.min()) / (network_2.max() - network_2.min())
# network_3 = (network_3 - network_3.min()) / (network_3.max() - network_3.min())

# if len(expt_names) == 4:
#     network_4 = (network_4 - network_4.min()) / (network_4.max() - network_4.min())

# if len(expt_names) == 5:
#     network_4 = (network_4 - network_4.min()) / (network_4.max() - network_4.min())
#     network_5 = (network_5 - network_5.min()) / (network_5.max() - network_5.min())

if plane == 'ax':
    if arguments.intervals == None:
        ax = [0, 512]
        ay = [0, 512]
    else:
        ax = [int(interval_list[0]), int(interval_list[1])]
        ay = [int(interval_list[2]), int(interval_list[3])]

    for idx in range(0, vol_dims[2]):
        hi_vol[:, :, idx] = (hi_vol[:, :, idx] - hi_vol[:, :, idx].min()) / (hi_vol[:, :, idx].max() - hi_vol[:, :, idx].min())
        lo_vol[:, :, idx] = (lo_vol[:, :, idx] - lo_vol[:, :, idx].min()) / (lo_vol[:, :, idx].max() - lo_vol[:, :, idx].min())
        int_vol[:, :, idx] = (int_vol[:, :, idx] - int_vol[:, :, idx].min()) / (int_vol[:, :, idx].max() - int_vol[:, :, idx].min())
        network_1[:, :, idx] = (network_1[:, :, idx] - network_1[:, :, idx].min()) / (network_1[:, :, idx].max() - network_1[:, :, idx].min())
        network_2[:, :, idx] = (network_2[:, :, idx] - network_2[:, :, idx].min()) / (network_2[:, :, idx].max() - network_2[:, :, idx].min())
        network_3[:, :, idx] = (network_3[:, :, idx] - network_3[:, :, idx].min()) / (network_3[:, :, idx].max() - network_3[:, :, idx].min())

        if len(expt_names) == 4:
            network_4[:, :, idx] = (network_4[:, :, idx] - network_4[:, :, idx].min()) / (network_4[:, :, idx].max() - network_4[:, :, idx].min())

        if len(expt_names) == 5:
            network_4[:, :, idx] = (network_4[:, :, idx] - network_4[:, :, idx].min()) / (network_4[:, :, idx].max() - network_4[:, :, idx].min())
            network_5[:, :, idx] = (network_5[:, :, idx] - network_5[:, :, idx].min()) / (network_5[:, :, idx].max() - network_5[:, :, idx].min())

        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Axial, slice {}'.format(idx))
        axs[0, 0].imshow(hi_vol[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
        
        # axs[0, 0].hist(hi_vol.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 0].set_title('Hi res')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(network_1[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
        # axs[0, 1].hist(lo_vol.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 1].set_title(expt_names[0])
        axs[0, 1].axis('off')
        
        axs[0, 2].imshow(network_2[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
        # axs[0, 2].hist(int_vol.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[0, 2].set_title(expt_names[1])
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(network_3[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
        # axs[1, 1].hist(network_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        axs[1, 0].set_title(expt_names[2])
        axs[1, 0].axis('off')

        if len(expt_names) == 3:
            axs[1, 1].imshow(lo_vol[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 1].hist(network_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 1].set_title('Lo res')
            axs[1, 1].axis('off')
            axs[1, 2].imshow(int_vol[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 2].set_title('Cubic interp')
            axs[1, 2].axis('off')

        elif len(expt_names) == 4:
            axs[1, 1].imshow(network_4[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 1].hist(network_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 1].set_title(expt_names[3])
            axs[1, 1].axis('off')
            axs[1, 2].imshow(int_vol[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 2].set_title('Cubic interp')
            axs[1, 2].axis('off')
        else:
            axs[1, 1].imshow(network_4[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_3.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 1].set_title(expt_names[3])
            axs[1, 1].axis('off')
            axs[1, 2].imshow(network_5[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_3.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 2].set_title(expt_names[4])
            axs[1, 2].axis('off')
        
        # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(np.std(int_vol[:, :, idx]), np.std(network_1[:, :, idx]),
        # np.std(network_2[:, :, idx]), np.std(network_3[:, :, idx]),
        # np.std(network_4[:, :, idx]), np.std(network_5[:, :, idx])))

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

if plane == 'co':
    if arguments.intervals == None:
        cx = [0, vol_dims[0]]
    else:
        cx = [int(interval_list[0]), int(interval_list[1])]

    cy = [0, vol_dims[0]]

    for idx in range(0, vol_dims[0], 8):  
        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Coronal, slice {}'.format(idx))
        axs[0, 0].imshow(hi_vol[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
        axs[0, 0].set_title('Hi res')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(network_1[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
        axs[0, 1].set_title(expt_names[0])
        axs[0, 1].axis('off')
        
        axs[0, 2].imshow(network_2[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
        axs[0, 2].set_title(expt_names[1])
        axs[0, 2].axis('off')

        axs[1, 0].imshow(network_3[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
        axs[1, 0].set_title(expt_names[2])
        axs[1, 0].axis('off')

        if len(expt_names) == 3:
            axs[1, 1].imshow(lo_vol[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
            # axs[1, 1].hist(network_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 1].set_title('Lo res')
            axs[1, 1].axis('off')
            axs[1, 2].imshow(int_vol[ax[0]:ax[1], ay[0]:ay[1], idx].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 2].set_title('Cubic interp')
            axs[1, 2].axis('off')

        elif len(expt_names) == 4:
            axs[1, 1].imshow(network_4[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
            # axs[1, 1].hist(network_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 1].set_title(expt_names[3])
            axs[1, 1].axis('off')
            axs[1, 2].imshow(int_vol[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
            # axs[1, 2].hist(network_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            axs[1, 2].set_title('Cubic interp')
            axs[1, 2].axis('off')
        else:
            axs[1, 1].imshow(network_4[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
            axs[1, 1].set_title(expt_names[3])
            axs[1, 1].axis('off')
            axs[1, 2].imshow(network_5[cx[0]:cx[1], idx, cy[0]:cy[1]].T, cmap='gray', origin='lower')
            axs[1, 2].set_title(expt_names[4])
            axs[1, 2].axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

if plane == 'sa':
    if arguments.intervals == None:
        sx = [0, vol_dims[1]]
    else:
        sx = [int(interval_list[0]), int(interval_list[1])]

    sy = [0, vol_dims[1]]

    for idx in range(0, vol_dims[1], 16):
        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Saggital, slice {}'.format(idx))
        axs[0, 0].imshow(hi_vol[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
        axs[0, 0].set_title('Hi res')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(network_1[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
        axs[0, 1].set_title(expt_names[0])
        axs[0, 1].axis('off')
        
        axs[0, 2].imshow(network_2[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
        axs[0, 2].set_title(expt_names[1])
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(network_3[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
        axs[1, 0].set_title(expt_names[2])
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(network_4[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
        axs[1, 1].set_title(expt_names[3])
        axs[1, 1].axis('off')

        if len(expt_names) != 5:
            axs[1, 2].imshow(int_vol[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
            axs[1, 2].set_title('Cubic interp')
            axs[1, 2].axis('off')
        else:
            axs[1, 2].imshow(network_4[idx, sx[0]:sx[1], sy[0]:sy[1]].T, cmap='gray', origin='lower')
            axs[1, 2].set_title(expt_names[4])
            axs[1, 2].axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()