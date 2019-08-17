@ECHO OFF
SET ana_path="C:\Users\rmappin\AppData\Local\Continuum\anaconda3\"
SET file_path="C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\006_CNN_3D_Super_Res"
SET save_path="C:\Users\rmappin\PhD_Data\Super_Res_Data\Toshiba_Vols\NPY_Test"
CALL %ana_path%\Scripts\activate.bat CpuEnv

SET list=nc4_ep50_n1026, nc4_ep50_n1026_fft1e1, nc4_ep50_n1026_fft3e1, nc4_ep50_n1026_fft1e2, nc4_ep50_n1026_fft3e2

FOR %%f IN (%list%) DO IF NOT EXIST %save_path%\%%f MKDIR %save_path%\%%f

FOR %%f IN (%list%) ^
DO ECHO ================================================ & ^
ECHO %%f & ^
CALL python %file_path%\scripts\output\Test_Image_Gen.py -ex %%f -mb 1 -nc 4

ECHO ================================================
PAUSE