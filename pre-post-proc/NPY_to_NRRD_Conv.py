from argparse import ArgumentParser
import nrrd
import numpy as np
import os


parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--subject', '-s', help="Subject number", type=str)
parser.add_argument('--vol_number', '-v', help="2nd volume start number", type=int)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase + '/'

FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NPY_Test/" + phase
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"
# FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase

if not os.path.exists(FILE_PATH):
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/test_data/"
    # FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/saved_images/" + phase
    pass

if arguments.subject == None:
    raise ValueError("Must provide subject number")
else:
    subject = arguments.subject

if arguments.vol_number == None:
    raise ValueError("Must provide 2nd volume start number")
else:
    vol = arguments.vol_number

out_list = os.listdir(FILE_PATH + expt_name + '/')
out_list.sort()

subject_list = [vol for vol in out_list if subject in vol]

SAVE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/Out/" + phase + expt_name + '/'
# SAVE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_data/temp/Out/" + expt_name + '/'
vol_dims = [512, 512, 12]

out_1 = np.zeros((512, 512, 12 * (vol - 1)))
out_2 = np.zeros((512, 512, 12 * (len(subject_list) - vol + 1)))

for i in range(vol - 1):
    out_vol = np.load(FILE_PATH + expt_name + '/' + subject_list[i])
    out_1[:, :, (i * 12):((i + 1) * 12)] = out_vol
    print("{}".format(subject_list[i]))

nrrd.write(os.path.join(SAVE_PATH, subject + '_1_1_O.nrrd'), out_1)
print("SAVED")

for i in range(len(subject_list) - vol + 1):
    out_vol = np.load(FILE_PATH + expt_name + '/' + subject_list[i + vol - 1])
    out_2[:, :, (i * 12):((i + 1) * 12)] = out_vol
    print("{}".format(subject_list[i + vol - 1]))

nrrd.write(os.path.join(SAVE_PATH, subject + '_1_2_O.nrrd'), out_2)
print("SAVED")
