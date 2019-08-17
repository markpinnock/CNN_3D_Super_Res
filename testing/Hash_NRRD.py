import hashlib
import nrrd
import numpy as np
import os

FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/NRRD_Test/"
SAVE_PATH = FILE_PATH + 'Hashed/'

hi_list = ['UCLH_11700946_1_1_NeR_H.nrrd', 'UCLH_17138405_1_2_NeR_H.nrrd',
'UCLH_21093614_1_1_NeL_H.nrrd', 'UCLH_22239993_1_2_NeR_H.nrrd', 'UCLH_23160588_1_1_NeL_H.nrrd']

int_list = [img[:-6] + 'I.nrrd' for img in hi_list]
out_list = [img[:-10] + 'O.nrrd' for img in hi_list]

# for img in int_list:
#     img_path = FILE_PATH + 'Int/' + img
#     img_hash = hashlib.sha256(img_path.encode()).hexdigest()
#     vol, _ = nrrd.read(img_path)
#     nrrd.write(os.path.join(SAVE_PATH, img_hash + '.nrrd'), vol)
#     print(img, img_hash)

model_list = os.listdir(FILE_PATH + 'Out/')

for model in model_list:
    for img in out_list:
        img_path = FILE_PATH + 'Out/' + model + '/' + img
        img_hash = hashlib.sha256(img_path.encode()).hexdigest()
        vol, _ = nrrd.read(img_path)
        nrrd.write(os.path.join(SAVE_PATH, img_hash + '.nrrd'), vol)
        print(img, img_hash)