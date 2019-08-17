from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as sci


np.set_printoptions(precision=2)

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Name of experiment", type=str)
parser.add_argument('--phase', '-p', help="Expt phase", type=str, nargs='?', const='3', default='3')
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase + '/'

FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "cross-validation/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "cross-validation/"

zero_list = [file for file in os.listdir(FILE_PATH) if expt_name in file and 'cv' in file]
# param_list = [file for file in os.listdir(FILE_PATH + 'fft/') if expt_name in file and 'fft' in file and 'cv' in file]

hyper_param_list = ['4', '8', '16', '32']
# hyper_param_floats = [float(val[3:]) for val in hyper_param_list]

fft_array = np.zeros((len(hyper_param_list), 5))
N_folds = 5
N_val = 0

for file in zero_list:
    chan_str = file.split('_')[0][2:]
    chan_idx = hyper_param_list.index(chan_str)
    fold_idx = int(file.split('_')[3][-1:])

    with open(FILE_PATH + file, 'r') as input:
        lines = input.readlines()

        for line in lines:
            if 'N_val' in line:
                temp_line = line.split(' ')[2]
                N_val = int(temp_line[:-1])

            if 'Summed validation' in line:
                temp_line = line.split('[')[1]
                fft_array[chan_idx, fold_idx] = float(temp_line.split(']')[0]) / N_val

fft_medians = np.median(fft_array, axis=1)
print(fft_medians)
fft_errors = 2 * np.std(fft_array, axis=1) / np.sqrt(N_folds)
# plt.plot(hyper_param_floats, fft_array.mean(axis=1), 'kx')
# plt.errorbar(hyper_param_floats, fft_means, fft_errors, fmt='kx', ecolor='r', capthick=None, capsize=4)
plt.boxplot(fft_array.T)
plt.title(expt_name)
plt.xlabel('Number of channels')
plt.ylabel('Validation loss')
plt.xticks([1, 2, 3, 4], hyper_param_list)
F, p = sci.f_oneway(fft_array[0, :], fft_array[1, :], fft_array[2, :], fft_array[3, :])
print(F, p)
plt.show()


