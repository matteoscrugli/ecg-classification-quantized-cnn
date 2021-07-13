import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



session_train_list = [dI for dI in os.listdir('output/train/') if os.path.isdir(os.path.join('output/train/',dI))]
accuracy = []
file = []
for line in session_train_list:
    try:
        if ('NLRAV' in line or 'NSVFQ' in line) and not ('aug' in line):
            json_file = open(f'output/train/{line}/training_summary.json', 'r')
            json_data = json.load(json_file)
            accuracy.append(json_data['fixed_point_accuracy'])
            file.append(line)
            json_file.close()
    except:
        continue

# print(accuracy)

accuracy = np.array(accuracy)
file = np.array(file)
sort = np.argsort(accuracy)

accuracy = np.flip(accuracy[sort])
file = np.flip(file[sort])

color = ['tab:blue']*accuracy.size

for i, f in enumerate(file):
    if 'NSVFQ' in f:
        color[i] = '#20639B' #'firebrick'
        # if 'aug' in f:
        #     color[i] = 'maroon'
    if 'NLRAV' in f:
        color[i] = '#F6655C' #'teal'
        # if 'aug' in f:
        #     color[i] = 'darkslategray'




width = 0.9

plt.figure()
# plt.suptitle(f"Fixed accuracy")

plt.xlim([0.9,1])
yoffset = -0.04
p1 = plt.barh(np.arange(len(accuracy)), accuracy, width, color=color)
for i, acc in enumerate(accuracy):
    plt.text(acc, i+yoffset, f' {acc:.4f} ', color='white', va='center', ha='right') #, rotation=-90, weight="bold"
for i, f in enumerate(file):
    plt.text(0.9, i+yoffset,f' {f} ', color='white', va='center', ha='left')#, rotation=90, weight="bold"

plt.ylabel('Model')
# plt.xlabel('Threat score')
plt.xlabel('Accuracy')
plt.yticks([])

plt.legend((p1[0], p1[-1]), ('NLRAV', 'NSVFQ'), loc='upper right')

plt.tight_layout()
plt.show()
