from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as sci


np.set_printoptions(precision=2)

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Name of experiment", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase

FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "/cross-validation/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "/cross-validation/"

zero_list = [file for file in os.listdir(FILE_PATH) if expt_name in file and 'cv' in file]
param_list = [file for file in os.listdir(FILE_PATH + 'fft/') if expt_name in file and 'fft' in file and 'cv' in file]

hyper_param_list = ['fft0e0', 'fft1e1', 'fft3e1', 'fft1e2', 'fft3e2']
hyper_param_floats = [float(val[3:]) for val in hyper_param_list]

fft_array = np.zeros((len(hyper_param_list), len(zero_list)))
N_folds = len(zero_list)
N_val = 0

for file in zero_list:
    fold_idx = int(file.split('_')[3][-1:])

    with open(FILE_PATH + file, 'r') as input:
        lines = input.readlines()

        for line in lines:
            if 'N_val' in line:
                temp_line = line.split(' ')[2]
                N_val = int(temp_line[:-1])

            if 'Summed validation' in line:
                temp_line = line.split('[')[1]
                fft_array[0, fold_idx] = float(temp_line.split(']')[0]) / N_val

for file in param_list:
    fft_idx = hyper_param_list.index(file.split('_')[3])
    fold_idx = int(file.split('_')[4][-1:])

    with open(FILE_PATH + 'fft/' + file, 'r') as input:
        lines = input.readlines()

        for line in lines:
            if 'N_val' in line:
                temp_line = line.split(' ')[2]
                N_val = int(temp_line[:-1])

            if 'Summed validation' in line:
                temp_line = line.split('[')[1]
                fft_array[fft_idx, fold_idx] = float(temp_line.split(']')[0]) / N_val


fft_means = fft_array.mean(axis=1)
fft_errors = 2 * np.std(fft_array, axis=1) / np.sqrt(N_folds)
# plt.plot(hyper_param_floats, fft_array.mean(axis=1), 'kx')
# plt.errorbar(hyper_param_floats, fft_means, fft_errors, fmt='kx', ecolor='r', capthick=None, capsize=4)
plt.boxplot(fft_array.T)
plt.title(expt_name)
plt.xlabel('FFT lambda')
plt.ylabel('Validation loss')
plt.xticks([1, 2, 3, 4, 5], ['0', '10', '30', '100', '300'])
F, p = sci.f_oneway(fft_array[0, :], fft_array[1, :], fft_array[2, :], fft_array[3, :], fft_array[4, :])
print(F, p)
plt.show()


