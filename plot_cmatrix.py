import os
import argparse
import numpy as np
import pickle as pk
import seaborn as sn
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from scipy.interpolate import make_interp_spline, BSpline

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', required=True, dest='train', type=dir_path, help="training session path")
parser.add_argument('-f', '--file', dest='file', nargs='*', help="file to plot")
parser.add_argument('-c','--colprop', dest='colprop', default=2, help="columns proportion")
parser.add_argument('-r','--rowprop', dest='rowprop', default=1, help="row proportion")

args = parser.parse_args()

session_path = args.train
files = args.file

# with open(f'{session_path}/confusionmatrix_float.pickle', 'rb') as input_file:
#     cmatrix_float = pk.load(input_file)
with open(f'{session_path}/confusionmatrix_fixed.pickle', 'rb') as input_file:
    cmatrix = pk.load(input_file)

json_file = open(f'{session_path}/training_summary.json', 'r')
json_data = json.load(json_file)
dataset_name = json_data['dataset_name']
fully_2_outdim = json_data['fully_2_outdim']



data_names = ['100', '101', '102', '103', '104', '105', '106', '107',
          '108', '109', '111', '112', '113', '114', '115', '116',
          '117', '118', '119', '121', '122', '123', '124', '200',
          '201', '202', '203', '205', '207', '208', '209', '210',
          '212', '213', '214', '215', '217', '219', '220', '221',
          '222', '223', '228', '230', '231', '232', '233', '234']

if files == None:
    files = data_names



for file in files:
    print()
    print(f'File: {file}')
    print(cmatrix[data_names.index(file)])



subplot_col_prop = float(args.colprop)
subplot_row_prop = float(args.rowprop)

subplot_col = round(math.sqrt(len(files)/(subplot_col_prop*subplot_row_prop)) * subplot_col_prop)
if subplot_col < 1:
    subplot_col = 1
subplot_row = math.ceil(len(files)/subplot_col)

if subplot_col:
     if subplot_col > len(files):
         subplot_col = len(files)
else:
    subplot_col = 1

while (subplot_col*subplot_row - subplot_row) >= len(files):
    subplot_col -= 1



fig = plt.figure()
fig.suptitle(f"Fixed point confusion matrix")
for i, f in enumerate(files):
    ax1 = fig.add_subplot(subplot_row,subplot_col,i+1)
    ax1.set_title(f'{f}', fontsize=10)

    cmap = sn.cubehelix_palette(gamma= 8, start=1.4, rot=.55, dark=0.8, light=1, as_cmap=True)
    df_cm = pd.DataFrame(cmatrix[data_names.index(f)], index = [i for i in dataset_name], columns = [i for i in dataset_name])
    res = sn.heatmap(df_cm, annot=True, fmt='g', cmap = cmap, annot_kws={"fontsize":6}) # vmax=2000.0
    for _, spine in res.spines.items():
        spine.set_visible(True)

    plt.ylabel("Predicted label")
    plt.xlabel("True label")

fig.tight_layout()
plt.show()
