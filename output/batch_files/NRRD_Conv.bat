@ECHO OFF
SET ana_path="C:\Users\rmappin\AppData\Local\Continuum\anaconda3\"
SET file_path="C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\006_CNN_3D_Super_Res"
SET save_path="C:\Users\rmappin\PhD_Data\Super_Res_Data\Toshiba_Vols\NRRD_Test\Out\Phase_3"

CALL %ana_path%\Scripts\activate.bat base

SET list=nc4_ep50_n1026_UNet, nc4_ep50_n1026_UNet2, nc8_ep50_n1026_UNet, nc8_ep50_n1026_UNet2

FOR %%f IN (%list%) DO IF NOT EXIST %save_path%\%%f MKDIR %save_path%\%%f
FOR %%f IN (%list%) DO ^
CALL python %file_path%\scripts\pre-post-proc\NPY_to_NRRD_Conv.py -ex %%f -s UCLH_11700946 -v 10 & ^
CALL python %file_path%\scripts\pre-post-proc\NPY_to_NRRD_Conv.py -ex %%f -s UCLH_17138405 -v 19 & ^
CALL python %file_path%\scripts\pre-post-proc\NPY_to_NRRD_Conv.py -ex %%f -s UCLH_21093614 -v 10 & ^
CALL python %file_path%\scripts\pre-post-proc\NPY_to_NRRD_Conv.py -ex %%f -s UCLH_22239993 -v 21 & ^
CALL python %file_path%\scripts\pre-post-proc\NPY_to_NRRD_Conv.py -ex %%f -s UCLH_23160588 -v 11

REM PAUSE
REM shutdown.exe /s /t 60