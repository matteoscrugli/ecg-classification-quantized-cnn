import numpy as np
import pickle as pk
import argparse
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-e','--eval', dest='eval', required=True, nargs='*', help="evaluation session path")
parser.add_argument('-c','--colprop', dest='colprop', default=2, help="columns proportion")
parser.add_argument('-r','--rowprop', dest='rowprop', default=1, help="row proportion")
parser.add_argument('-s','--sharedaxes', dest='sharedaxes', action='store_true', help="shared axes")
parser.add_argument('-m','--median', dest='median', action='store_true', help="print median value")

args = parser.parse_args()

session_path = args.eval

tp_cmatrix_fixed = []
tp_cmatrix_fixed_centered = []
tp_cmatrix_float = []
tp_cmatrix_float_centered = []

i = 0
for path in session_path:


    with open(f'{path}/tp_cmatrix_float_centered.pickle', 'rb') as input_file:
        tp_cmatrix_float_centered.append([])
        tp_cmatrix_float_centered[i] = np.array(pk.load(input_file))

    with open(f'{path}/tp_cmatrix_fixed_centered.pickle', 'rb') as input_file:
        tp_cmatrix_fixed_centered.append([])
        tp_cmatrix_fixed_centered[i] = np.array(pk.load(input_file))

    with open(f'{path}/tp_cmatrix_float.pickle', 'rb') as input_file:
        tp_cmatrix_float.append([])
        tp_cmatrix_float[i] = np.array(pk.load(input_file))

    with open(f'{path}/tp_cmatrix_fixed.pickle', 'rb') as input_file:
        tp_cmatrix_fixed.append([])
        tp_cmatrix_fixed[i] = np.array(pk.load(input_file))

    i = i + 1

# exit()

scc = 0
data = []

for n in range(len(session_path)):
    data.append([])
    data[n].append([])
    for matrix in tp_cmatrix_float_centered[n]:
        if np.sum(matrix):
            data[n][0].append(np.sum(matrix.diagonal())/np.sum(matrix))

    data[n].append([])
    for matrix in tp_cmatrix_fixed_centered[n]:
        if np.sum(matrix):
            data[n][1].append(np.sum(matrix.diagonal())/np.sum(matrix))

    data[n].append([])
    for matrix in tp_cmatrix_float[n]:
        if np.sum(matrix):
            data[n][2].append(np.sum(matrix.diagonal())/np.sum(matrix))

    data[n].append([])
    for matrix in tp_cmatrix_float[n]:
        if np.sum(matrix):
            data[n][3].append(np.sum(matrix.diagonal())/np.sum(matrix))



subplot_col_prop = float(args.colprop)
subplot_row_prop = float(args.rowprop)
shared_axes = args.sharedaxes
median_value = args.median

subplot_col = round(math.sqrt(len(session_path)/(subplot_col_prop*subplot_row_prop)) * subplot_col_prop)
if subplot_col < 1:
    subplot_col = 1
subplot_row = math.ceil(len(session_path)/subplot_col)

if subplot_col:
     if subplot_col > len(session_path):
         subplot_col = len(session_path)
else:
    subplot_col = 1

while (subplot_col*subplot_row - subplot_row) >= len(session_path):
    subplot_col -= 1



fig = plt.figure()

# fig.suptitle(f"Peaks evaluation's box plot")
for i, (d, p) in enumerate(zip(data, session_path)):
    if i and shared_axes:
        ax1 = fig.add_subplot(subplot_row,subplot_col,i+1, sharey=ax1)
    else:
        ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)
        # ax1.set_ylabel('Threat score')
        ax1.set_ylabel('Accuracy')

    i and ax1.set_ylabel('Accuracy')

    if 'NLRAV' in p:
        if 'aug' in p:
            ax1.set_title(f'NLRAV, with augmentation')
        else:
            ax1.set_title(f'NLRAV, without augmentation')

    elif 'NSVFQ' in p:
        if 'aug' in p:
            ax1.set_title(f'NSVFQ, with augmentation')
        else:
            ax1.set_title(f'NSVFQ, without augmentation')

    ax1.boxplot(d)
    # ax1.set_xlabel("", color=color)
    # plt.xticks([1, 2, 3, 4], ['float_cen', 'fixed_cen', 'float', 'fixed'])
    plt.xticks([1, 2, 3, 4], ['floating point\ncenentered', 'fixed point\ncentered', 'floating point', 'fixed point'])
    d = np.array(d)
    if median_value:
        for x , y in enumerate(d):
            plt.text(x+1, 1, f'{np.median(np.array(y)):.4f}', va='bottom', ha='center', fontsize=10, color='orange', weight='bold')#.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='orange'))

plt.show()
