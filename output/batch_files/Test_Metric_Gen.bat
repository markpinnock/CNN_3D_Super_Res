@ECHO OFF
SET ana_path="C:\Users\rmappin\AppData\Local\Continuum\anaconda3\"
SET file_path="C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\006_CNN_3D_Super_Res"
SET save_path="C:\Users\rmappin\PhD_Data\Super_Res_Data\Toshiba_Vols\Metric_Test\Phase_3"

CALL %ana_path%\Scripts\activate.bat base

SET list=nc4_ep20_n1026, nc4_ep20_n1026_fft1e1, nc4_ep20_n1026_fft3e1, nc4_ep20_n1026_fft1e2, nc4_ep20_n1026_fft3e2,^
nc8_ep20_n1026, nc8_ep20_n1026_fft1e1, nc8_ep20_n1026_fft3e1, nc8_ep20_n1026_fft1e2, nc8_ep20_n1026_fft3e2,^
nc16_ep20_n1026, nc16_ep20_n1026_fft1e1, nc16_ep20_n1026_fft3e1, nc16_ep20_n1026_fft1e2, nc16_ep20_n1026_fft3e2,^
nc32_ep10_n1026, nc32_ep10_n1026_fft1e1, nc32_ep10_n1026_fft3e1, nc32_ep10_n1026_fft1e2, nc32_ep10_n1026_fft3e2

FOR %%f IN (%list%) DO IF NOT EXIST %save_path%\%%f MKDIR %save_path%\%%f
FOR %%f IN (%list%) DO ^
CALL python %file_path%\scripts\output\Test_Metrics.py -ex %%f

PAUSE
REM shutdown.exe /s /t 60