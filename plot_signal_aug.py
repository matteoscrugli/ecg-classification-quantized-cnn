import wfdb
import os
import argparse
import shutil
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('-f','--files', dest='files', required=True, nargs='*', help="ecg recordings file name")
parser.add_argument('-p','--position', dest='position', nargs='*', type=int, help="position (multimple input works only for one registration at a time)")
parser.add_argument('-s','--size', dest='size', default=1000, type=int, help="frame half size")
parser.add_argument('-a','--augmentation', dest='augmentation', nargs=2, type=int, default=[0,1], help='augmentation, number of lateral shifts and pitch (two arguments)')
parser.add_argument('-c','--colprop', dest='colprop', default=1, help="columns proportion")
parser.add_argument('-r','--rowprop', dest='rowprop', default=4, help="row proportion")

args = parser.parse_args()



files = args.files

labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']

sig = []
lab = []
pos = []
lab_filtered = []
pos_filtered = []
sig_filtered = []
th = []
for i, file in enumerate(files):
    f_th = open('./output/threshold/default/'+file+'_th.txt', 'r')
    th.append(int(f_th.read()))

    lab_filtered.append([])
    pos_filtered.append([])
    r = wfdb.rdrecord('./dataset/raw/'+file)
    ann = wfdb.rdann('./dataset/raw/'+file, 'atr', return_label_elements=['label_store', 'symbol'])
    sig.append(np.array(r.p_signal[:,0]))
    # intsig = np.array(r.p_signal[:,0])
    # sig_len = len(sig)
    lab.append(ann.symbol)
    pos.append(ann.sample)

    for l, p in zip(lab[i], pos[i]):
        if l in labels:
            lab_filtered[i].append(l)
            pos_filtered[i].append(p)

    sig_filtered.append([])
    f_sig_filtered = open('./output/dataset/raw_text/'+file+'_filtered.txt', 'r')
    Lines = f_sig_filtered.readlines()
    for line in Lines:
        sig_filtered[i].append(int(line.strip()))
    f_sig_filtered.close()
    # os.remove('./output/dataset/raw_text/'+d+'_filtered.txt')

f_filter_delay = open('./output/dataset/raw_text/filter_delay.txt', 'r')
filter_delay = int(f_filter_delay.read())



subplot_col_prop = float(args.colprop)
subplot_row_prop = float(args.rowprop)
position = []

if args.position != None:
    position.extend(args.position)
if args.size != None:
    size = args.size

num_files = len(files)
num_plot = len(position) if len(files) == 1 else len(files)
num_plot = num_plot if num_plot else 1

subplot_col = round(math.sqrt(num_plot/(subplot_col_prop*subplot_row_prop)) * subplot_col_prop)
if subplot_col < 1:
    subplot_col = 1
subplot_row = math.ceil(num_plot/subplot_col)

if subplot_col:
     if subplot_col > num_plot:
         subplot_col = num_plot
else:
    subplot_col = 1

while (subplot_col*subplot_row - subplot_row) >= num_plot:
    subplot_col -= 1



augmentation = args.augmentation

fig = plt.figure()
# fig.suptitle(f"ECG raw & filtered signal")

for i, P in enumerate(position):
    ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)

    color = 'black'
    ax1.plot(range(P-size,P+size),np.array(sig[0][P-size:P+size]), color=color)
    # ax1.set(title=f'File: {file}')
    # ax1.set(title=f'Augmentation')
    ax1.set_ylabel("Raw signal") #, color=color
    ax1.set_xlabel("Sample") #, color=color
    ax1.grid()

    for j in range(-augmentation[0]*augmentation[1],augmentation[0]*augmentation[1]+1,augmentation[1]):
        color = (random.random(), random.random(), random.random())
        linewidth = 2.5
        # alpha = 1
        alpha = 1-((abs(j)/(augmentation[0]*augmentation[1]))*0.9)
        # alpha = 1-((j-(-augmentation[0]*augmentation[1]))/(2*augmentation[0]*augmentation[1])*0.9)
        ax1.plot(range(P-99+j,P+99+j), [min(sig[0][P-size:P+size]) - (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1]*(99*2), linewidth=linewidth, color=color, alpha=alpha)
        ax1.plot(range(P-99+j,P+99+j), [max(sig[0][P-size:P+size]) + (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1]*(99*2), linewidth=linewidth, color=color, alpha=alpha)
        ax1.plot([P-0.0000001-99+j,P+0.0000001-99+j], [min(sig[0][P-size:P+size]) - (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1 , max(sig[0][P-size:P+size]) + (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1], linewidth=linewidth, color=color, alpha=alpha)
        ax1.plot([P-0.0000001+99+j,P+0.0000001+99+j], [min(sig[0][P-size:P+size]) - (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1 , max(sig[0][P-size:P+size]) + (max(sig[0][P-size:P+size]) - min(sig[0][P-size:P+size]))*0.1], linewidth=linewidth, color=color, alpha=alpha)

fig.tight_layout()
plt.show()
