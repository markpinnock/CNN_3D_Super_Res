from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os


# np.set_printoptions(precision=2)

parser = ArgumentParser()
parser.add_argument('--expt_name', '-ex', help="Name of experiment", type=str)
parser.add_argument('--phase', '-p', help='Expt phase', type=str, nargs='?', const='3', default='3')
parser.add_argument('--individual', '-i', help="Individual fold plots", type=str, nargs='?', const='n', default='n')
arguments = parser.parse_args()

if arguments.expt_name == None:
    raise ValueError("Must provide experiment name")
else:
    expt_name = arguments.expt_name

phase = 'Phase_' + arguments.phase +'/'

FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "/cross-validation/"

if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/006_CNN_3D_Super_Res/reports/" + phase + "/cross-validation/"

if arguments.individual == 'y':
    ind_flag = True
else:
    ind_flag = False

if 'fft' in expt_name:
    FILE_PATH = FILE_PATH + 'fft/'

file_list = os.listdir(FILE_PATH)
report_list = [file for file in file_list if expt_name in file and 'cv' in file]

report_split = report_list[0].split('_')
epochs = list(range(int(report_split[1][2:])))

if 'fft' in report_list[0]:
    LAMBDA = report_split[4][3:]
else:
    LAMBDA = 0

summed_validation_losses = 0
N_val = 0
N_folds = len(report_list)
total_training_losses = np.zeros((len(epochs)))
total_validation_losses = np.zeros((len(epochs)))

main_ax = plt.subplot(111)

for report in report_list:
    fold_training_losses = []
    fold_validation_losses = []

    with open(FILE_PATH + report, 'r') as report:
        lines = report.readlines()

        for line in lines:

            if 'train loss' in line:
                fold_training_losses.append(float(line.split(' ')[4]))

            if 'val loss' in line:
                fold_validation_losses.append(float(line.split(' ')[4]))

            if 'Summed validation' in line:
                summed_validation_losses += float(line.split(' ')[6])
            
            if 'N_val' in line:
                N_val += int(line.split(' ')[2])
                # N_val += int(temp_line[:-1])

        total_training_losses += fold_training_losses
        total_validation_losses += fold_validation_losses
        print("Train/val losses: {} {}".format(fold_training_losses[-1:], fold_validation_losses[-1:]))

        if ind_flag == True:
            sub_ax = plt.subplot(111)
            sub_ax.plot(epochs, fold_training_losses, 'k')
            sub_ax.plot(epochs, fold_validation_losses, 'r')
            plt.show()

if ind_flag == False:
    main_ax.plot(epochs, total_training_losses / N_folds, 'k')
    main_ax.plot(epochs, total_validation_losses / N_folds, 'r')
    main_ax.plot((epochs[0], epochs[-1]), (summed_validation_losses / N_val, summed_validation_losses / N_val), 'k--')
    # main_ax.set_title('Expt ' + title_num + ', ' +\
    #     num_chan + ' channels' + ', ' + num_data + ' pairs' +\
    #         ', ' + 'lambda ' + str(float(LAMBDA)))
    main_ax.set_title(report_list[0][:-4])
    main_ax.set_xlabel('Epochs')
    main_ax.set_ylabel('Loss per volume')
    main_ax.set_ylim(bottom=0)
    # main_ax.legend(('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Val loss: {:.2f}'.format(val_losses / int(num_data))))
    main_ax.legend(('Training loss', 'Validation loss', 'Val loss: {}'.format(summed_validation_losses / N_val)))
    plt.show()
