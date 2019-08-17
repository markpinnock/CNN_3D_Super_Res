import hashlib
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import scipy.stats as sci
import sys

sys.path.append('..')

from utils.losses import calcDice


FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/"
HASH_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/Hashed/"

hi_list = ['UCLH_11700946_1_1_NeR_H.nrrd', 'UCLH_17138405_1_2_NeR_H.nrrd',
'UCLH_21093614_1_1_NeL_H.nrrd', 'UCLH_22239993_1_2_NeR_H.nrrd', 'UCLH_23160588_1_1_NeL_H.nrrd']

int_list = [img[:-6] + 'I.nrrd' for img in hi_list]
out_list = [img[:-10] + 'O.nrrd' for img in hi_list]

hi_list.sort()
int_list.sort()
out_list.sort()

num_subs = 5
num_mods = 20

temp_dice = np.zeros(num_subs)
all_dice = np.zeros((num_subs, num_mods + 1))

for idx, img in enumerate(int_list):
    hi_path = FILE_PATH + 'Hi/' + hi_list[idx]
    hi_hash = hashlib.sha256(hi_path.encode()).hexdigest()
    hi_hash = hi_hash + '.seg.nrrd'
    int_path = FILE_PATH + 'Int/' + img
    int_hash = hashlib.sha256(int_path.encode()).hexdigest()
    int_hash = int_hash + '.seg.nrrd'

    if int_hash in os.listdir(HASH_PATH) and hi_hash in os.listdir(HASH_PATH):
        hi_seg, _ = nrrd.read(HASH_PATH + hi_hash)
        hi_seg = hi_seg[0, ...]
        int_seg, _ = nrrd.read(HASH_PATH + int_hash)
        int_seg = int_seg[0, ...]
        temp_dice[idx] = calcDice(hi_seg, int_seg)

    all_dice[:, 0] = temp_dice

# model_list = os.listdir(FILE_PATH + 'Out/Phase_2/')
# model_list.sort()
model_list = ['nc4_ep20_n1026', 'nc4_ep20_n1026_fft1e1', 'nc4_ep20_n1026_fft3e1', 'nc4_ep20_n1026_fft1e2', 'nc4_ep20_n1026_fft3e2',
 'nc8_ep20_n1026', 'nc8_ep20_n1026_fft1e1', 'nc8_ep20_n1026_fft3e1', 'nc8_ep20_n1026_fft1e2', 'nc8_ep20_n1026_fft3e2',
 'nc16_ep20_n1026', 'nc16_ep20_n1026_fft1e1', 'nc16_ep20_n1026_fft3e1', 'nc16_ep20_n1026_fft1e2', 'nc16_ep20_n1026_fft3e2',
 'nc32_ep10_n1026', 'nc32_ep10_n1026_fft1e1', 'nc32_ep10_n1026_fft3e1', 'nc32_ep10_n1026_fft1e2', 'nc32_ep10_n1026_fft3e2']

model_xticks = ['Interp', 'nc4, lmda = 0', 'nc4, lmda = 10', 'nc4, lmda = 30', 'nc4, lmda = 100', 'nc4, lmda = 300',
'nc8, lmda = 0', 'nc8, lmda = 10', 'nc8, lmda = 30', 'nc8, lmda = 100', 'nc8, lmda = 300',
'nc16, lmda = 0', 'nc16, lmda = 10', 'nc16, lmda = 30', 'nc16, lmda = 100', 'nc16, lmda = 300',
'nc32, lmda = 0', 'nc32, lmda = 10', 'nc32, lmda = 30', 'nc32, lmda = 100', 'nc32, lmda = 300']

for model_idx, model in enumerate(model_list):
#     if 'nc4_ep20_n1026' == model:
#         model_idx = 1
#         print(model, model_idx)

#     if 'nc8_ep20_n1026' == model:
#         model_idx = 6
#         print(model, model_idx)

#     if 'nc16_ep20_n1026' == model:
#         model_idx = 11
#         print(model, model_idx)

#     if 'nc32_ep20_n1026' == model:
#         model_idx = 16
#         print(model, model_idx)

#     if '_n1026_' in model:
#         model_idx += 1
#         print(model, model_idx)

    for idx, img in enumerate(out_list):
        hi_path = FILE_PATH + 'Hi/' + hi_list[idx]
        hi_hash = hashlib.sha256(hi_path.encode()).hexdigest()
        hi_hash = hi_hash + '.seg.nrrd'
        out_path = FILE_PATH + 'Out/' + model + '/' + img
        out_hash = hashlib.sha256(out_path.encode()).hexdigest()
        out_hash = out_hash + '.seg.nrrd'

        if out_hash in os.listdir(HASH_PATH) and hi_hash in os.listdir(HASH_PATH):
            hi_seg, _ = nrrd.read(HASH_PATH + hi_hash)
            hi_seg = hi_seg[0, ...]
            out_seg, _ = nrrd.read(HASH_PATH + out_hash)
            out_seg = out_seg[0, ...]
            temp_dice[idx] = calcDice(hi_seg, out_seg)

        all_dice[:, model_idx + 1] = temp_dice

int_dice = all_dice[:, 0]
sub_dice = all_dice[:, 1:21:5]
sub_dice = np.hstack((int_dice[:, np.newaxis], sub_dice))

fig, ax = plt.subplots(1, 1)
ax.boxplot(sub_dice)
ax.set_xticklabels([model_xticks[0]] + model_xticks[1:21:5], rotation=40)

F_dice, p_dice = sci.f_oneway(*sub_dice.T)
print(sub_dice.mean(axis=0))
print([sub_dice.mean(axis=0) - (2*np.std(sub_dice, axis=0)/np.sqrt(5)), sub_dice.mean(axis=0) + (2*np.std(sub_dice, axis=0)/np.sqrt(5))])
print("F = {:.2f}, p = {:.2f}".format(F_dice, p_dice))

plt.show()
