from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

file_path = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/CNN_3D_Super_Res/reports/"
file_list = os.listdir(file_path)

parser = ArgumentParser()
parser.add_argument('--expt_number', '-ex', help="Number of experiment", type=str)
arguments = parser.parse_args()

if arguments.expt_number == None:
    raise ValueError("Must provide experiment number")
else:
    expt_num = arguments.expt_number

report_list = [file for file in file_list if expt_num in file and 'nc' in file]
report_split = report_list[0].split('_')
title_num = report_split[1]
num_chan = report_split[2][2:]
epochs = list(range(int(report_split[3][2:])))
num_data = report_split[4][1:]
val_losses = 0

for report in report_list:
    training_losses = []

    with open(file_path + report, 'r') as report:
        lines = report.readlines()

        for line in lines:
            if 'Epoch' in line:
                training_losses.append(float(line.split('[')[1][:-2]))
        
            if 'Summed' in line:
                val_losses += float(line.split('[')[1][:-2])

        plt.plot(epochs, training_losses)

plt.plot((epochs[0], epochs[-1]), (val_losses / int(num_data), val_losses / int(num_data)), 'k--')
plt.title('Expt ' + title_num + ', ' + num_chan + ' channels' + ', ' + num_data + ' pairs')
plt.xlabel('Epochs')
plt.ylabel('Loss per volume')
plt.ylim(bottom=0)
plt.legend(('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Val loss: {:.2f}'.format(val_losses / int(num_data))))
plt.show()
            