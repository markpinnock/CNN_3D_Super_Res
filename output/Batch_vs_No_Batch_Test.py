import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as sci
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


def reportReader(report, validation, pSNR, SSIM, N_val):

    with open(FILE_PATH + report, 'r') as report:
        lines = report.readlines()
        
        for line in lines:
            if 'Summed validation' in line:
                validation.append(float(line.split(' ')[6]))
            
            if 'SSIM' in line:
                SSIM.append(float(line.split(' ')[3]))

            if 'pSNR' in line:
                pSNR.append(float(line.split(' ')[3]))

            if 'N_val' in line:
                N_val.append(int(line.split(' ')[2]))


FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_logs/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/test_logs/"

file_list = os.listdir(FILE_PATH)
reports = [file for file in file_list]

N_val = []
no_N_val = []
N_val2 = []
no_N_val2 = []
epochs = list(range(20))
batch_validation = []
no_batch_validation = []
batch_pSNR = []
no_batch_pSNR = []
batch_SSIM = []
no_batch_SSIM = []
batch_validation2 = []
no_batch_validation2 = []
batch_pSNR2 = []
no_batch_pSNR2 = []
batch_SSIM2 = []
no_batch_SSIM2 = []

for report in reports:
    if 'TEST_BATCH_' in report:
        reportReader(report, batch_validation, batch_pSNR, batch_SSIM, N_val)
    
    if 'TEST_NO_BATCH_' in report:
        reportReader(report, no_batch_validation, no_batch_pSNR, no_batch_SSIM, no_N_val)

    if 'TEST_BATCH2' in report:
        reportReader(report, batch_validation2, batch_pSNR2, batch_SSIM2, N_val2)
    
    if 'TEST_NO_BATCH2' in report:
        reportReader(report, no_batch_validation2, no_batch_pSNR2, no_batch_SSIM2, no_N_val2)

for N, loss in zip(N_val, batch_validation):
    loss *= N / sum(N_val)

for N, loss in zip(no_N_val, no_batch_validation):
    loss *= N / sum(no_N_val)

for N, loss in zip(N_val2, batch_validation2):
    loss *= N / sum(N_val2)

for N, loss in zip(no_N_val2, no_batch_validation2):
    loss *= N / sum(no_N_val2)

fig, axs = plt.subplots(1, 3)
axs[0].boxplot([batch_validation, no_batch_validation, batch_validation2, no_batch_validation2])
axs[1].boxplot([batch_pSNR, no_batch_pSNR, batch_pSNR2, no_batch_pSNR2])
axs[2].boxplot([batch_SSIM, no_batch_SSIM, batch_SSIM2, no_batch_SSIM2])
axs[0].set_xticklabels(['UNet BN', 'UNet no BN', 'UNet2 BN', 'UNet2 no BN'], rotation=40)
axs[1].set_xticklabels(['UNet BN', 'UNet no BN', 'UNet2 BN', 'UNet2 no BN'], rotation=40)
axs[2].set_xticklabels(['UNet BN', 'UNet no BN', 'UNet2 BN', 'UNet2 no BN'], rotation=40)

F_MSE, p_MSE = sci.f_oneway(batch_validation, no_batch_validation, batch_validation2, no_batch_validation2)
print("sig: F = {:.2f}, p = {:.2f}".format(F_MSE, p_MSE))
F_pSNR, p_pSNR = sci.f_oneway(batch_pSNR, no_batch_pSNR, batch_pSNR2, no_batch_pSNR2)
print("sig: F = {:.2f}, p = {:.2f}".format(F_pSNR, p_pSNR))
F_SSIM, p_SSIM = sci.f_oneway(batch_SSIM, no_batch_SSIM, batch_SSIM2, no_batch_SSIM2)
print("sig: F = {:.2f}, p = {:.2f}".format(F_SSIM, p_SSIM))

print(np.median(batch_validation), np.median(no_batch_validation), np.median(batch_validation2), np.median(no_batch_validation2))
print(np.median(batch_pSNR), np.median(no_batch_pSNR), np.median(batch_pSNR2), np.median(no_batch_pSNR2))
print(np.median(batch_SSIM), np.median(no_batch_SSIM), np.median(batch_SSIM2), np.median(no_batch_SSIM2))

group_nums = np.array(list(range(4)))
group_nums = np.repeat(group_nums, 5)

tukey_MSE = MultiComparison(batch_validation + no_batch_validation + batch_validation2 + no_batch_validation2, group_nums)
res_MSE = tukey_MSE.tukeyhsd()
tukey_pSNR = MultiComparison(batch_pSNR + no_batch_pSNR + batch_pSNR2 + no_batch_pSNR2, group_nums)
res_pSNR = tukey_pSNR.tukeyhsd()
tukey_SSIM = MultiComparison(batch_SSIM + no_batch_SSIM + batch_SSIM2 + no_batch_SSIM2, group_nums)
res_SSIM = tukey_SSIM.tukeyhsd()
 
print("MSE Tukey")
print("========================")
print(res_MSE)

print("pSNR Tukey")
print("========================")
print(res_pSNR)

print("SSIM Tukey")
print("========================")
print(res_SSIM)

plt.show()
