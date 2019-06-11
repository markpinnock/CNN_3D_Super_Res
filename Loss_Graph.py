import matplotlib.pyplot as plt
import os

file_path = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD Prog/CNN_3D_Super_Res/reports/"
file_list = os.listdir(file_path)
report_num = 0

report_name = file_list[report_num]
epochs = list(range(100))

training_losses = []

with open(file_path + report_name, 'r') as report:
    lines = report.readlines()

    for line in lines:
        if 'Epoch' in line:
            training_losses.append(float(line.split('[')[1][:-2]))

plt.plot(epochs, training_losses)
plt.show()
            