from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as sciint
import scipy.stats as sci
import skimage.measure as sk
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Experiment name", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--lo_int', '-li', help="2 for both, 1 for int, 0 for none", type=int, nargs='?', const=2, default=2)
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_string = arguments.expt_name
    expt_names = expt_string.split(' ')

phase = 'Phase_' + arguments.phase + '/'

if arguments.lo_int < 0 or arguments.lo_int > 2:
    raise ValueError("2 for both, 1 for int, 0 for none")
else:
    num_extra = arguments.lo_int

FILE_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/Metric_Test/"
NET_PATH = "C:/Users/rmappin/PhD_Data/Super_Res_Data/Toshiba_Vols/Metric_Test/" + phase

lo_MSE = np.load(FILE_PATH + 'lo_MSE.npy')
lo_pSNR = np.load(FILE_PATH + 'lo_pSNR.npy')
lo_SSIM = np.load(FILE_PATH + 'lo_SSIM.npy')

int_MSE = np.load(FILE_PATH + 'int_MSE.npy')
int_pSNR = np.load(FILE_PATH + 'int_pSNR.npy')
int_SSIM = np.load(FILE_PATH + 'int_SSIM.npy')

MSE_array = np.zeros((len(lo_MSE), len(expt_names) + num_extra))
pSNR_array = np.zeros((len(lo_pSNR), len(expt_names) + num_extra))
SSIM_array = np.zeros((len(lo_SSIM), len(expt_names) + num_extra))

if num_extra == 2:
    MSE_array[:, 0] = lo_MSE.T
    MSE_array[:, 1] = int_MSE.T
    pSNR_array[:, 0] = lo_pSNR.T
    pSNR_array[:, 1] = int_pSNR.T
    SSIM_array[:, 0] = lo_SSIM.T
    SSIM_array[:, 1] = int_SSIM.T

if num_extra == 1:
    MSE_array[:, 1] = int_MSE.T
    pSNR_array[:, 1] = int_pSNR.T
    SSIM_array[:, 1] = int_SSIM.T

for idx, expt_name in enumerate(expt_names):
    METRIC_SAVE_PATH = FILE_PATH + expt_name + '/'

    out_MSE = np.load(NET_PATH + expt_name + '/' + expt_name + '_MSE.npy')
    out_pSNR = np.load(NET_PATH + expt_name  + '/' + expt_name+ '_pSNR.npy')
    out_SSIM = np.load(NET_PATH + expt_name  + '/' + expt_name+ '_SSIM.npy')

    MSE_array[:, idx + num_extra] = out_MSE.T
    pSNR_array[:, idx + num_extra] = out_pSNR.T
    SSIM_array[:, idx + num_extra] = out_SSIM.T

fig, axs = plt.subplots(1, 3)

axs[0].boxplot(MSE_array)
axs[0].set_title("MSE")
axs[1].boxplot(pSNR_array)
axs[1].set_title("pSNR")
axs[2].boxplot(SSIM_array)
axs[2].set_title("SSIM")

if num_extra == 2:
    # axs[0].set_xticklabels(['Lo Res', 'Interp'] + expt_names, rotation=0)
    axs[0].set_xticklabels(['Lo Res', 'Interp'] + expt_names, rotation=40)
    axs[1].set_xticklabels(['Lo Res', 'Interp'] + expt_names, rotation=40)

if num_extra == 1:
    # axs[0].set_xticklabels(['Interp'] + expt_names, rotation=40)
    axs[0].set_xticklabels(['Interp'] + expt_names, rotation=40)
    axs[1].set_xticklabels(['Interp'] + expt_names, rotation=40)

if num_extra == 0:
    # axs[0].set_xticklabels(expt_names, rotation=40)
    axs[0].set_xticklabels(expt_names, rotation=40)
    axs[1].set_xticklabels(expt_names, rotation=40)

F_MSE, p_MSE = sci.f_oneway(*MSE_array.T)
print("Mean MSE {}, sig: F = {:.2f}, p = {:.2f}".format(np.mean(MSE_array, axis=0), F_MSE, p_MSE))
F_pSNR, p_pSNR = sci.f_oneway(*pSNR_array.T)
print("Mean pSNR {}, sig: F = {:.2f}, p = {:.2f}".format(np.mean(pSNR_array, axis=0), F_pSNR, p_pSNR))
F_SSIM, p_SSIM = sci.f_oneway(*SSIM_array.T)
print("Median SSIM {}, sig: F = {:.2f}, p = {:.2f}".format(np.median(SSIM_array, axis=0), F_SSIM, p_SSIM))

group_nums = np.array(list(range(len(expt_names) + num_extra)))
group_nums = np.repeat(group_nums, len(lo_MSE))

tukey_MSE = MultiComparison(MSE_array.T.ravel(), group_nums)
res_MSE = tukey_MSE.tukeyhsd()
tukey_pSNR = MultiComparison(pSNR_array.T.ravel(), group_nums)
res_pSNR = tukey_pSNR.tukeyhsd()
tukey_SSIM = MultiComparison(SSIM_array.T.ravel(), group_nums)
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
