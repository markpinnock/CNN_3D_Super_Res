@ECHO OFF
SET ana_path="C:\Users\rmappin\AppData\Local\Continuum\anaconda3\"
SET file_path="C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\006_CNN_3D_Super_Res"
ECHO Test Metrics > metrics.txt

CALL %ana_path%\Scripts\activate.bat base

FOR /D %%f IN (%file_path%\models\Phase_2\*) ^
DO ECHO ================================================ >> metrics.txt & ^
ECHO %%~nf & ^
ECHO %%~nf >> metrics.txt & ^
CALL python -W ignore %file_path%\scripts\output\Test_Metrics.py -ex %%~nxf >> metrics.txt
ECHO ================================================ >> metrics.txt
PAUSE