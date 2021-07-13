import wfdb
import os
import argparse
import shutil
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('-f','--files', dest='files', required=True, nargs='*', help="ecg recordings file name")
parser.add_argument('-p','--position', dest='position', nargs='*', type=int, help="position (multimple input works only for one registration at a time)")
parser.add_argument('-s','--size', dest='size', default=1000, type=int, help="frame half size")
parser.add_argument('-c','--colprop', dest='colprop', default=1, help="columns proportion")
parser.add_argument('-r','--rowprop', dest='rowprop', default=4, help="row proportion")
parser.add_argument('-fl','--filtered', dest='filtered', action='store_true', help="plot also the filtered signal")

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




fig = plt.figure()
# fig.suptitle(f"ECG raw & filtered signal")

if num_files != 1:
    for i, (s, s_filtered, file) in enumerate(zip(sig, sig_filtered, files)):
        ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)



        color = 'tab:blue'
        if position:
            ax1.plot(range(position[0]-size,position[0]+size),np.array(s[position[0]-size:position[0]+size]), color=color)
        else:
            ax1.plot(np.array(s), color=color)
        ax1.set(title=f'File: {file}')
        ax1.set_ylabel("Raw signal", color=color)
        ax1.set_xlabel("Sample")
        ax1.grid()

        for p, l in enumerate(lab_filtered[i]):
            if len(position):
                if pos_filtered[i][p] >= position[0]-size and pos_filtered[i][p] <= position[0]+size:
                    plt.text(pos_filtered[i][p],s[pos_filtered[i][p]],f'{lab_filtered[i][p]}', va='bottom', ha='center', weight="bold")
            else:
                plt.text(pos_filtered[i][p],s[pos_filtered[i][p]],f'{lab_filtered[i][p]}', va='bottom', ha='center', weight="bold")


        if args.filtered:
            color = 'tab:red'
            ax2 = ax1.twinx()
            ax2.set_ylabel("Filtered signal", color=color, alpha=0.5)

            if len(position):
                ax2.plot(range(position[0]-size,position[0]+size),s_filtered[position[0]-size+filter_delay:position[0]+size+filter_delay], color=color, alpha=0.45)
                ax2.plot(range(position[0]-size,position[0]+size),[th[i]]*(size*2), linestyle='dashed', color=color, alpha=0.35)
                plt.text(position[0]+size,th[i], 'Threshold', color=color, alpha=0.65, weight="bold", va='bottom', ha='right')
            else:
                ax2.plot(s_filtered[filter_delay:], color=color, alpha=0.45)
                ax2.plot([th[i]]*(len(s_filtered[filter_delay:])), linestyle='dashed', color=color, alpha=0.35)
                plt.text(len(s_filtered[filter_delay:]),th[i], 'Threshold', color=color, alpha=0.65, weight="bold", va='bottom', ha='right')

else:
    if len(position):
        for i, P in enumerate(position):
            ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)



            color = 'tab:blue'
            ax1.plot(range(P-size,P+size),np.array(sig[0][P-size:P+size]), color=color)
            ax1.set(title=f'File: {file}')
            ax1.set_ylabel("Raw signal", color=color)
            ax1.grid()

            for x,y in zip(range(2*size), np.array(sig[0][P-size:P+size])):
                print(f'{x} {y}')

            for p, l in enumerate(lab_filtered[0]):
                if pos_filtered[0][p] >= P-size and pos_filtered[0][p] <= P+size:
                    plt.text(pos_filtered[0][p], sig[0][pos_filtered[0][p]],f'{lab_filtered[0][p]}', va='bottom', ha='center', weight="bold")

            ax1.set_xticks([])
            ax1.set_yticks([])


            if args.filtered:
                color = 'tab:red'
                ax2 = ax1.twinx()
                ax2.set_ylabel("Filtered signal", color=color, alpha=0.5)
                ax2.plot(range(P-size,P+size), sig_filtered[0][P-size+filter_delay:P+size+filter_delay], color=color, alpha=0.45)
                ax2.plot(range(P-size,P+size), [th[0]]*(size*2), linestyle='dashed', color=color, alpha=0.35)
                plt.text(P+size,th[i], 'Threshold', color=color, alpha=0.65, weight="bold", va='bottom', ha='right')
    else:
        ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)



        color = 'tab:blue'
        if position:
            ax1.plot(range(P-size,P+size),np.array(sig[0][P-size:P+size]), color=color)
        else:
            ax1.plot(np.array(sig[0]), color=color)
        ax1.set(title=f'File: {file}')
        ax1.set_ylabel("Raw signal", color=color)
        ax1.grid()

        for p, l in enumerate(lab_filtered[0]):
            plt.text(pos_filtered[0][p],sig[0][pos_filtered[0][p]],f'{lab_filtered[0][p]}', va='bottom', ha='center', weight="bold")


        if args.filtered:
            color = 'tab:red'
            ax2 = ax1.twinx()
            ax2.set_ylabel("Filtered signal", color=color, alpha=0.5)

            ax2.plot(sig_filtered[0][filter_delay:], color=color, alpha=0.45)
            ax2.plot([th[0]]*(len(sig_filtered[0][filter_delay:])), linestyle='dashed', color=color, alpha=0.35)
            plt.text(len(sig_filtered[0][filter_delay:]),th[i], 'Threshold', color=color, alpha=0.65, weight="bold", va='bottom', ha='right')

fig.tight_layout()
plt.show()
