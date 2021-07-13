import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



models = ['4_4_100',    '20_20_25',     '20_4_100',     '20_20_100',    'NLRAV_20_20_10',    'NSVFQ_4_4_25']
energy = [148.78,       591.93,         476.27,         660.37,         560.65,         120.14]
classes = ['NLRAV', 'NSVFQ']
threshold_pvalue = 0.005
threshold = [0, 0]
colors = ['#F6655C', '#20639B']
markers = ['^', 'o']
orientation_ver = ['bottom', 'top']
orientation_hor = ['center', 'center']
rotation = [90,90]
plot_ylimit = [0.975,1]
font_size=10
marker_size = 80
alpha_th = 0.25



session_train_list = [dI for dI in os.listdir('output/train/') if os.path.isdir(os.path.join('output/train/',dI))]
accuracy = [[0]*len(models) for i in classes]
file = [[0]*len(models) for i in classes]



for line in session_train_list:
    try:
        if any(m in line for m in models) and not 'aug' in line:
            for i, c in enumerate(classes):
                if c in line:
                    json_file = open(f'output/train/{line}/training_summary.json', 'r')
                    json_data = json.load(json_file)
                    accuracy[i][[idx for idx, m in enumerate(models) if m in line][0]] = json_data['fixed_point_accuracy']
                    file[i][[idx for idx, m in enumerate(models) if m in line][0]] = line.replace(f'{c}_','')
                    json_file.close()
    except:
        continue

for i, acc in enumerate(accuracy):
    threshold[i] = max(acc)-threshold_pvalue



plt.grid(color='lightgray',linestyle='--', alpha=0.6, zorder=0)
for acc, col, mar, cla, thr, rot, fil, ver, hor in zip(accuracy, colors, markers, classes, threshold, rotation, file, orientation_ver, orientation_hor):
    plt.scatter(energy, acc, c=col, marker=mar, s=marker_size, label=cla, zorder=2)
    plt.plot([plt.xlim()[1]-max(energy),max(energy)],[thr,thr], linestyle='dashed', c=col, alpha=alpha_th, zorder=1)
    for e, a, f in zip(energy, acc, fil):
        if a < thr:
            plt.scatter(e, a, c=col, marker='x', s=marker_size*5, label=None, zorder=2) #, label=cla
        plt.text(e, a, f'   {f}   ', va=ver, ha=hor, rotation=rot, fontsize=font_size, color=col, weight="bold", zorder=3)



plt.xlim(left = 0)
plt.ylim(plot_ylimit)
plt.xlabel("Energy (uJ)")
# plt.ylabel("Threat score")
plt.ylabel("Accuracy")
plt.legend(loc='lower left')
plt.show()
