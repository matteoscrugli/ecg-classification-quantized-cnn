import torch
import torchvision
import torch.quantization
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pk
import pandas as pd
import wfdb
import pywt
#import h5py
import math
import os
import sys
import argparse
from pathlib import Path
import shutil
import copy
import time
import json

import itertools
import threading

from torch.quantization import QuantStub, DeQuantStub
from pathlib import Path
from torch.utils import data

class color:
    NONE = ''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

parser = argparse.ArgumentParser()

parser.add_argument('-n','--name', dest='name', required=True, help="session name")
parser.add_argument('-e','--epoch', dest='epoch', required=True, type=int, help="number of epochs")
parser.add_argument('-d','--dataset', dest='dataset', required=True, choices=['NLRAV', 'NSVFQ', 'NLRAVU', 'NSVFQU', 'NSV', 'SV'], help="choice of dataset between NLRAV or NSVFQ")
parser.add_argument('-s','--split', dest='split', default='0.7', help="choice of dataset splitting")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")
parser.add_argument('-b','--batchsize', dest='batchsize', default=32, type=int, help="batchsize value")
parser.add_argument('-a','--augmentation', dest='augmentation', nargs=2, type=int, default=[0,1], help='augmentation, number of lateral shifts and pitch (two arguments)')
parser.add_argument('-r','--randomseed', dest='randomseed', type=int, default=0, help='random seed for dataset randomization')
parser.add_argument('-p','-.peak', dest='peak', help='peak detector path')
parser.add_argument('--norm', dest='normalization', action='store_true', help="during training, scales all inputs so that its absolute value is equal to 1")
parser.add_argument('--indim', dest='indimension', default=198, type=int, help="input dimension")
parser.add_argument('--ksize', dest='ksize', default=7, type=int, help="kernel size")
parser.add_argument('--conv1of', dest='conv1of', default=20, type=int, help="conv 1 output features value")
parser.add_argument('--conv2of', dest='conv2of', default=20, type=int, help="conv 2 output features value")
parser.add_argument('--foutdim', dest='foutdim', default=100, type=int, help="fully connected 1 output dimension")

args = parser.parse_args()



session_name = args.name
session_path = "output/train/"+session_name+"/"
if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
            Path(session_path+'inference_data_example').mkdir(parents=True, exist_ok=True)
            Path(session_path+'parameters').mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in session creation ("+session_path+").")
            exit()
    else:
        # print("Session already exists ("+session_path+"), overwrite the session? (y/n): ", end='')
        # force_write = input()
        # if force_write == "y":
        #     print('')
        #     try:
        #         shutil.rmtree(session_path)
        #         Path(session_path).mkdir(parents=True, exist_ok=True)
        #         Path(session_path+'inference_data_example').mkdir(parents=True, exist_ok=True)
        #         Path(session_path+'parameters').mkdir(parents=True, exist_ok=True)
        #     except OSError:
        #         print("Error in session creation ("+session_path+").")
        #         exit()
        # else:
        #     exit()
        print(f'Session path ({session_path}) already exists')
        exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
        Path(session_path+'inference_data_example').mkdir(parents=True, exist_ok=True)
        Path(session_path+'parameters').mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

print(f'{color.BOLD}Starting {color.NONE}training{color.END}{color.BOLD} session \'{session_name}\'\n\n\n{color.END}')









#██╗   ██╗ █████╗ ██████╗ ██╗ ██████╗ ██╗   ██╗███████╗
#██║   ██║██╔══██╗██╔══██╗██║██╔═══██╗██║   ██║██╔════╝
#██║   ██║███████║██████╔╝██║██║   ██║██║   ██║███████╗
#╚██╗ ██╔╝██╔══██║██╔══██╗██║██║   ██║██║   ██║╚════██║
# ╚████╔╝ ██║  ██║██║  ██║██║╚██████╔╝╚██████╔╝███████║
#  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝  ╚═════╝ ╚══════╝

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_size_of_model(model):
    torch.save(model.state_dict(), session_path+"temp.p")
    print('Size model (MB):', os.path.getsize(session_path+"temp.p")/1e6)
    os.remove('temp.p')

def save_model(model, n):
    torch.save(model.state_dict(), session_path+"model"+n+".pth")

t_done = False
def animate(prefix = ''):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if t_done:
            break
        print('\r' + prefix + c, end = '\r')
        # sys.stdout.write('\r' + prefix + c)
        # sys.stdout.flush()
        time.sleep(0.2)
    print('\r' + prefix + 'Done!')
    # sys.stdout.write('\r' + prefix + 'Done!')

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print(f'\r{prefix} |{bar}| Done! {suffix}')








# ██╗  ██╗██████╗ ███████╗███████╗    ██╗      ██████╗  █████╗ ██████╗ ███████╗██████╗
# ██║  ██║██╔══██╗██╔════╝██╔════╝    ██║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
# ███████║██║  ██║█████╗  ███████╗    ██║     ██║   ██║███████║██║  ██║█████╗  ██████╔╝
# ██╔══██║██║  ██║██╔══╝  ╚════██║    ██║     ██║   ██║██╔══██║██║  ██║██╔══╝  ██╔══██╗
# ██║  ██║██████╔╝██║     ███████║    ███████╗╚██████╔╝██║  ██║██████╔╝███████╗██║  ██║
# ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚══════╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝

# class HDF5Dataset(data.Dataset):
#     """Represents an abstract HDF5 dataset.
#
#     Input params:
#         file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
#         recursive: If True, searches for h5 files in subdirectories.
#         load_data: If True, loads all the data immediately into RAM. Use this if
#             the dataset is fits into memory. Otherwise, leave this at false and
#             the data will load lazily.
#         data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
#         transform: PyTorch transform to apply to every data instance (default=None).
#     """
#     def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
#         super().__init__()
#         self.data_info = []
#         self.data_cache = {}
#         self.data_cache_size = data_cache_size
#         self.transform = transform
#
#         # Search for all h5 files
#         p = Path(file_path)
#         assert(p.is_dir())
#         if recursive:
#             files = sorted(p.glob('**/*.h5'))
#         else:
#             files = sorted(p.glob('*.h5'))
#         if len(files) < 1:
#             raise RuntimeError('No hdf5 datasets found')
#
#         for h5dataset_fp in files:
#             self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
#
#     def __getitem__(self, index):
#         # get data
#         x = self.get_data("X_data", index)
# #        print(x)
#         x = np.expand_dims(x,axis=1)
#         if self.transform:
#             x = self.transform(x)
#         else:
#             x = torch.from_numpy(x)
#
#         # get label
#         y = self.get_data("y_data", index)
# #        print(y)
# #        exit()
#         y = torch.from_numpy(y)
#         y=y.max(dim=1)[1]
#         return (x, y)
#
#     def __len__(self):
#         return len(self.get_data_infos('X_data'))
#
#     def _add_data_infos(self, file_path, load_data):
#         with h5py.File(file_path, 'r') as h5_file:
#             # Walk through all groups, extracting datasets
#             for dname, ds in h5_file.items():
#                 # if data is not loaded its cache index is -1
#                 idx = -1
#                 if load_data:
#                     # add data to the data cache
# #                    idx = self._add_to_cache(ds.value, file_path)
#                     idx = self._add_to_cache(ds[()], file_path)
#
#                 # type is derived from the name of the dataset; we expect the dataset
#                 # name to have a name such as 'data' or 'label' to identify its type
#                 # we also store the shape of the data in case we need it
# #                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})
#                 self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})
#
#     def _load_data(self, file_path):
#         """Load data to the cache given the file
#         path and update the cache index in the
#         data_info structure.
#         """
#
#         with h5py.File(file_path, 'r') as h5_file:
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():
#                     # add data to the data cache and retrieve
#                     # the cache index
#                     idx = self._add_to_cache(ds.value, file_path)
#
#                     # find the beginning index of the hdf5 file we are looking for
#                     file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
#
#                     # the data info should have the same index since we loaded it in the same way
#                     self.data_info[file_idx + idx]['cache_idx'] = idx
#
#         # remove an element from data cache if size was exceeded
#         if len(self.data_cache) > self.data_cache_size:
#             # remove one item from the cache at random
#             removal_keys = list(self.data_cache)
#             removal_keys.remove(file_path)
#             self.data_cache.pop(removal_keys[0])
#             # remove invalid cache_idx
#             self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]
#
#     def _add_to_cache(self, data, file_path):
#         """Adds data to the cache and returns its index. There is one cache
#         list for every file_path, containing all datasets in that file.
#         """
#         if file_path not in self.data_cache:
#             self.data_cache[file_path] = [data]
#         else:
#             self.data_cache[file_path].append(data)
#         return len(self.data_cache[file_path]) - 1
#
#     def get_data_infos(self, type):
#         """Get data infos belonging to a certain type of data.
#         """
#         data_info_type = [di for di in self.data_info if di['type'] == type]
#         return data_info_type
#
#     def get_data(self, type, i):
#         """Call this function anytime you want to access a chunk of data from the
#             dataset. This will make sure that the data is loaded in case it is
#             not part of the data cache.
#         """
#         fp = self.get_data_infos(type)[i]['file_path']
#         if fp not in self.data_cache:
#             self._load_data(fp)
#
#         # get new cache_idx assigned by _load_data_info
#         cache_idx = self.get_data_infos(type)[i]['cache_idx']
#         return self.data_cache[fp][cache_idx]









#██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
#██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
#██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
#██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
#██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
#╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝

dataset_name = args.dataset
dataset_split = float(args.split)
value_batch_size = args.batchsize
normalization = args.normalization
augmentation = args.augmentation
random_seed = args.randomseed



# dataset = HDF5Dataset('output/dataset/NLRAVpeaks_hdf5/', recursive=True, load_data=True, data_cache_size=1, transform=None)
#
# for x, y in dataset:
#     t_dataset = torch.utils.data.TensorDataset(x,y)
#
# item_perm = torch.randperm(int(x.size(0)))
#
# x=x[item_perm]
# y=y[item_perm]
#
# X_train = x[:int(round(x.size(0)*value_dataset_split,0))]
# X_valid = x[int(round(x.size(0)*value_dataset_split,0)):]
# Y_train = y[0:int(round(x.size(0)*value_dataset_split,0))]
# Y_valid = y[int(round(x.size(0)*value_dataset_split,0)):]
#
# X_train.unsqueeze_(1)
# X_valid.unsqueeze_(1)






# if dataset_name == 'NLRAV' and args.split=='0.7':
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NLRAV/training_set_NLRAV_set_data.pickle", "rb") as input_file:
#         X_train = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NLRAV/training_set_NLRAV_labels.pickle", "rb") as input_file:
#         Y_train = pk.load(input_file)
#
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NLRAV/validation_set_NLRAV_set_data.pickle", "rb") as input_file:
#         X_valid = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NLRAV/validation_set_NLRAV_labels.pickle", "rb") as input_file:
#         Y_valid = pk.load(input_file)
#
# if dataset_name == 'NLRAV' and args.split=='0.8':
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NLRAV/training_set_NLRAV_set_data.pickle", "rb") as input_file:
#         X_train = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NLRAV/training_set_NLRAV_labels.pickle", "rb") as input_file:
#         Y_train = pk.load(input_file)
#
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NLRAV/validation_set_NLRAV_set_data.pickle", "rb") as input_file:
#         X_valid = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NLRAV/validation_set_NLRAV_labels.pickle", "rb") as input_file:
#         Y_valid = pk.load(input_file)
#
# if dataset_name == 'NSVFQ' and args.split=='0.7':
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NSVFQ/training_set_NSVFQ_set_data.pickle", "rb") as input_file:
#         X_train = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NSVFQ/training_set_NSVFQ_labels.pickle", "rb") as input_file:
#         Y_train = pk.load(input_file)
#
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NSVFQ/validation_set_NSVFQ_set_data.pickle", "rb") as input_file:
#         X_valid = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_70_30_NSVFQ/validation_set_NSVFQ_labels.pickle", "rb") as input_file:
#         Y_valid = pk.load(input_file)
#
# if dataset_name == 'NSVFQ' and args.split=='0.8':
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NSVFQ/training_set_NSVFQ_set_data.pickle", "rb") as input_file:
#         X_train = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NSVFQ/training_set_NSVFQ_labels.pickle", "rb") as input_file:
#         Y_train = pk.load(input_file)
#
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NSVFQ/validation_set_NSVFQ_set_data.pickle", "rb") as input_file:
#         X_valid = pk.load(input_file)
#     with open("./output/dataset/raw_peaks_aug/shuffled_80_20_NSVFQ/validation_set_NSVFQ_labels.pickle", "rb") as input_file:
#         Y_valid = pk.load(input_file)






X = []
Y = []
C = []
R = []
P = []



data_names = ['100', '101', '102', '103', '104', '105', '106', '107',
              '108', '109', '111', '112', '113', '114', '115', '116',
              '117', '118', '119', '121', '122', '123', '124', '200',
              '201', '202', '203', '205', '207', '208', '209', '210',
              '212', '213', '214', '215', '217', '219', '220', '221',
              '222', '223', '228', '230', '231', '232', '233', '234']

if dataset_name == 'NLRAV':
    labels = ['N', 'L', 'R', 'A', 'V']
    sub_labels = {'N':'N', 'L':'L', 'R':'R', 'A':'A', 'V':'V'}

elif dataset_name == 'NSVFQ':
    labels = ['N', 'S', 'V', 'F', 'Q']
    sub_labels = { 'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
                   'A':'S', 'a':'S', 'J':'S', 'S':'S',
                   'V':'V', 'E':'V',
                   'F':'F',
                   '/':'Q', 'f':'Q', 'Q':'Q'}

elif dataset_name == 'NSV':
    labels = ['N', 'S', 'V']
    sub_labels = { 'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
                   'A':'S', 'a':'S', 'J':'S', 'S':'S',
                   'V':'V', 'E':'V'}

elif dataset_name == 'SV':
    labels = ['S', 'V']
    sub_labels = { 'A':'S', 'a':'S', 'J':'S', 'S':'S',
                   'V':'V', 'E':'V'}

elif dataset_name == 'NLRAVU':
    labels = ['N', 'L', 'R', 'A', 'V', 'U']
    sub_labels = {'N':'N', 'L':'L', 'R':'R', 'A':'A', 'V':'V', 'U':'U'}

elif dataset_name == 'NSVFQU':
    labels = ['N', 'S', 'V', 'F', 'Q', 'U']
    sub_labels = { 'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
                   'A':'S', 'a':'S', 'J':'S', 'S':'S',
                   'V':'V', 'E':'V',
                   'F':'F',
                   '/':'Q', 'f':'Q', 'Q':'Q',
                   'U':'U'}



half_window = 99



# if 'U' in labels:
#     peak_path = args.peak
#     with open(f'{peak_path}/matrix_l.pickle', 'rb') as input_file:
#         matrix_l = pk.load(input_file)

# peak_path = args.peak
# with open(f'{peak_path}/matrix_l.pickle', 'rb') as input_file:
#     matrix_l = pk.load(input_file)

printProgressBar(0, len(data_names), prefix = 'Dataset building:', suffix = '', length = 50)
for d in data_names:
    r = wfdb.rdrecord('./dataset/raw/'+d)
    ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    sig = np.array(r.p_signal[:,0])
    intsig = np.array(r.p_signal[:,0])
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample

    # if 'U' in labels:
    #     sym_len = len(sym)
    #     for i in range(0, sym_len, 4):
    #         if int(pos[i] + pos[i+1]) > 100:
    #             sym.append('U')
    #             pos = np.append(pos,[int((pos[i] + pos[i+1]) / 2)])

    # if 'U' in labels:
    #     for matrix, i in zip(matrix_l[data_names.index(d)], range(len(matrix_l[data_names.index(d)]))):
    #         if len(matrix) == 0:
    #             sym.append('U')
    #             pos = np.append(pos,pos[i])

    # if d == '231':
    #     for i, matrix in enumerate(matrix_l[data_names.index(d)]):
    #         if len(matrix) == 0:
    #                 print(pos[i])
    #     exit()

    beat_len = len(sym)
    for i in range(beat_len):
        for j in range(-augmentation[0]*augmentation[1],augmentation[0]*augmentation[1]+1,augmentation[1]):
            if pos[i]-half_window+j>=0 and pos[i]+half_window+j<=sig_len and sym[i] in sub_labels:
                frame = sig[pos[i]-half_window+j:pos[i]+half_window+j]
                X.append(frame)
                Y.append(labels.index(sub_labels[sym[i]]))
                C.append(True if j == 0 else False)
                R.append(data_names.index(d))
                # P.append(pos[i])
    printProgressBar(data_names.index(d) + 1, len(data_names), prefix = 'Dataset building:', suffix = '', length = 50)



t_done = False
t_dict = {'prefix' : f'{color.NONE}Data loader{color.END}: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

item_perm = np.arange(np.size(X,0))

np.random.seed(random_seed)
np.random.shuffle(item_perm)

X = np.array(X)[item_perm]
Y = np.array(Y)[item_perm]
C = np.array(C)[item_perm]
R = np.array(R)[item_perm]
# P = np.array(P)[item_perm]

X_train = X[:round(np.size(X,0)*dataset_split)]
Y_train = Y[:round(np.size(X,0)*dataset_split)]
# C_train = C[:round(np.size(X,0)*dataset_split)]
# R_train = R[:round(np.size(X,0)*dataset_split)]
# P_train = P[:round(np.size(X,0)*dataset_split)]

X_valid = X[round(np.size(X,0)*dataset_split):]
Y_valid = Y[round(np.size(X,0)*dataset_split):]
C_valid = C[round(np.size(X,0)*dataset_split):]
R_valid = R[round(np.size(X,0)*dataset_split):]
# P_valid = P[round(np.size(X,0)*dataset_split):]

X_valid = X_valid[C_valid]
Y_valid = Y_valid[C_valid]
# R_valid = R_valid[C_valid]
# P_valid = P_valid[C_valid]






if normalization:
    for i in range(np.size(X_train,0)):
        X_train[i]=X_train[i]/np.max(np.absolute(X_train[i]))
    for i in range(np.size(X_valid,0)):
        X_valid[i]=X_valid[i]/np.max(np.absolute(X_valid[i]))






X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
Y_train = torch.from_numpy(Y_train)
Y_valid = torch.from_numpy(Y_valid)

X_train.unsqueeze_(1)
X_train.unsqueeze_(1)
X_valid.unsqueeze_(1)
X_valid.unsqueeze_(1)



t_dataset_train = torch.utils.data.TensorDataset(X_train,Y_train)
t_dataset_valid = torch.utils.data.TensorDataset(X_valid,Y_valid)

loader_train = torch.utils.data.DataLoader(t_dataset_train, batch_size=value_batch_size, shuffle=False)
loader_valid = torch.utils.data.DataLoader(t_dataset_valid, batch_size=value_batch_size, shuffle=False)

t_done = True
time.sleep(0.2)
print('\n\n')









#███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
#████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
#██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
#██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
#██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
#╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

conv_indim = args.indimension

pool_ks = 2

conv_1_if = 1
conv_1_of = args.conv1of
conv_1_ks = args.ksize

conv_2_if = conv_1_of
conv_2_of = args.conv2of
conv_2_ks = args.ksize

fully_1_indim = int(conv_2_of * ((((conv_indim - (conv_1_ks - 1)) / pool_ks) -(conv_1_ks - 1)) / pool_ks))
fully_1_outdim = args.foutdim

fully_2_indim = fully_1_outdim
fully_2_outdim = len(labels)



class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        self.relu6 = False
        self.debug = False
        self.quantization = False
        self.quantization_inf = False
        self.temp = 0

        self.minoutput_0 = 0
        self.maxoutput_0 = 0

        self.conv1 = nn.Conv2d(conv_1_if, conv_1_of, (1, conv_1_ks), bias=False)
        self.conv2 = nn.Conv2d(conv_2_if, conv_2_of, (1, conv_2_ks), bias=False)

        self.pool = nn.MaxPool2d((1, pool_ks))

        self.fc1 = nn.Linear(fully_1_indim, fully_1_outdim, bias=False)
        self.fc2 = nn.Linear(fully_2_indim, fully_2_outdim, bias=False)

        self.sm = nn.Softmax(dim=-1)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):

        if(self.debug):
            torch.set_printoptions(threshold=500000, precision=10) #, linehalf_windowth=20
            f = open(session_path+"inference_data_example/input_"+str(self.temp)+".txt", "w")

            f.write("\n\ndequant\n")
            f.write(str(x))

        if(self.quantization):
            x = self.quant(x)

        if(self.debug):
            f.write("\n\nquant\n")
            f.write(str(x))

        x = self.conv1(x)

        if(self.quantization_inf):
            if(torch.min(x)<self.minoutput_0):
                self.minoutput_0 = torch.min(x)
            if(torch.max(x)>self.maxoutput_0):
                self.maxoutput_0 = torch.max(x)

        if(self.debug):
            f.write("\n\nconv1\n")
            f.write(str(x))

        x = F.relu6(x)

        if(self.debug):
            f.write("\n\nrelu1\n")
            f.write(str(x))

        x = self.pool(x)

        if(self.debug):
            f.write("\n\npool1\n")
            f.write(str(x))



        x = self.conv2(x)

        if(self.debug):
            f.write("\n\nconv2\n")
            f.write(str(x))

        x = F.relu6(x)

        if(self.debug):
            f.write("\n\nrelu2\n")
            f.write(str(x))

        x = self.pool(x)

        if(self.debug):
            f.write("\n\npool2\n")
            f.write(str(x))


        x = x.flatten(1)

        if(self.debug):
            f.write("\n\nflatten\n")
            f.write(str(x))


        x=self.fc1(x)

        if(self.debug):
            f.write("\n\nfc1\n")
            f.write(str(x))

        x = F.relu6(x)

        if(self.debug):
            f.write("\n\nrelu3\n")
            f.write(str(x))


        x = self.fc2(x)

        if(self.debug):
            f.write("\n\nfc2\n")
            f.write(str(x))


        if(self.quantization):
            x = self.dequant(x)

        if(self.debug):
            f.write("\n\ndequant\n\n")
            f.write(str(x))
            f.close()


#        x = self.sm(x)

        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)









#████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

num_trainepoch = args.epoch
num_trainepoch_effective = 0
dim_batches = 25



model = Net()

# optimizer = optim.Adam(model.parameters(), lr=0.0005)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

criterion = nn.CrossEntropyLoss()



max_i = []

train_data = [[] for i in range(5)]
train_dic = {
    'train_loss' : 0,
    'valid_loss' : 1,
    'train_acc' : 2,
    'valid_acc' : 3,
    'learing_rate' : 4
}

epoch_loss = 0
epoch_acc = 0
cnt_allbatches = 0
tmp_cnt = 0
tmp_cnt_t = 0
frac = 33

print(len(loader_train))

print('\n\n\n\n\n', end = '')
printProgressBar(cnt_allbatches, len(loader_train)/dim_batches * num_trainepoch, prefix = f'{color.NONE}Training:{color.END}', suffix = '', length = 55)
print('\033[F\033[F\033[F\033[F\033[F', end = '')
try:
    for epoch in range(num_trainepoch):  # loop over the dataset multiple times

        cnt_batches = 0
        cnt = 0
        cnt_t = 0
        epoch_loss = 0

        running_loss = 0.0

        printProgressBar(0, len(loader_train), prefix = 'Epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '              ', length = 40)
        for i, data in enumerate(loader_train):


            inputs, labels = data

            print(len(inputs))
            exit()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)

            list.clear(max_i)
            for o in outputs:
                m=max(o)
                indx=list(o).index(m)
                max_i.append(indx)

            for o, m in zip(labels, max_i):
                if o == m:
                    cnt = cnt + 1
                cnt_t = cnt_t + 1

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i == len(loader_train) - 1:
                epoch_loss = epoch_loss/len(loader_train)

            running_loss += loss.item()
            if (i % dim_batches) == (dim_batches - 1):
                printProgressBar(i, len(loader_train) - 1, prefix = 'Session \''+session_name+'\', epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '                 ', length = 25)
                if i == len(loader_train) - 1:
                    print('', end='\033[F')
                    # training_loss.append(epoch_loss)
                print('\nLoss during training: %f\nAccuracy during training: %f\n\n' % (epoch_loss/i, (cnt/cnt_t)))
                printProgressBar(cnt_allbatches + i, len(loader_train) * num_trainepoch - 1, prefix = 'Training:', suffix = '', length = 55)
                if cnt_allbatches + i == (len(loader_train) * num_trainepoch) - 1:
                    print('', end='\033[F')
                print('\033[F\033[F\033[F\033[F\033[F\033[F')
                running_loss = 0.0
            elif i == (len(loader_train) - 1):
                # training_loss.append(epoch_loss)
                printProgressBar(i, len(loader_train) - 1, prefix = 'Session \''+session_name+'\', epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '                 ', length = 25)
                print('', end='\033[F')
                print('\nLoss during training: %f\nAccuracy during training: %f\n\n' % (epoch_loss, (cnt/cnt_t)))
                printProgressBar(cnt_allbatches + i, len(loader_train) * num_trainepoch - 1, prefix = 'Training:', suffix = '', length = 55)
                if cnt_allbatches + i == (len(loader_train) * num_trainepoch) - 1:
                    print('', end='\033[F')
                print('', end='\033[F\033[F')
                running_loss = 0.0

        cnt_allbatches = cnt_allbatches + len(loader_train)

        train_data[train_dic['learing_rate']].append(optimizer.param_groups[0]['lr'])

        train_data[train_dic['train_acc']].append(cnt/cnt_t)
        train_data[train_dic['train_loss']].append(epoch_loss)


        cnt = 0
        cnt_t = 0
        epoch_loss = 0

        # print(optimizer.param_groups[0]['lr'])

        t_done = False
        t_dict = {'prefix' : 'Accuracy on validation set: '}
        t = threading.Thread(target=animate, kwargs=t_dict)
        t.start()
        for i, data in enumerate(loader_valid):

            inputs, labels = data
            outputs = model(inputs.float())

            list.clear(max_i)
            for o in outputs:
                m=max(o)
                indx=list(o).index(m)
                max_i.append(indx)


            for o, m in zip(labels, max_i):
                if o == m:
                    cnt = cnt + 1
                cnt_t = cnt_t + 1

            epoch_loss += loss.item()
            if i == len(loader_valid) - 1:
                epoch_loss = epoch_loss/len(loader_valid)

        t_done = True
        time.sleep(0.2)

        train_data[train_dic['valid_acc']].append(cnt/cnt_t)
        train_data[train_dic['valid_loss']].append(epoch_loss)

        epoch_acc = cnt/cnt_t
        # training_acc.append(epoch_acc)
        num_trainepoch_effective += 1
        print('\033[FAccuracy on validation set: %f\n' % epoch_acc)
        printProgressBar(cnt_allbatches, len(loader_train) * num_trainepoch, prefix = f'{color.NONE}Training{color.END}:', suffix = '', length = 55)
except KeyboardInterrupt:
    print('\n\n\n\n\n')
print('\n')

save_model(model,'')









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#  ╚══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=7, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    topn = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        printProgressBar(0, len(data_loader), prefix = 'Calibrate:', suffix = '', length = 40)
        for image, target in data_loader:
            output = model(image.float())
            loss = criterion(output, target)
            cnt += 1
            acc1, accn = accuracy(output, target, topk=(1, fully_2_outdim))
            # print('.', end = '')
            top1.update(acc1[0], image.size(0))
            topn.update(accn[0], image.size(0))
            printProgressBar(cnt, len(data_loader), prefix = 'Calibrate:', suffix = '', length = 40)
            if cnt >= neval_batches:
                 return top1, topn

    return top1, top5

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('\nLoss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ██████╗  ██████╗ ███████╗████████╗████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ██████╔╝██║   ██║███████╗   ██║█████╗██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔═══╝ ██║   ██║╚════██║   ██║╚════╝██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██║     ╚██████╔╝███████║   ██║      ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
# ╚═══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝      ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

model_quantized = copy.deepcopy(model)
model_quantized.quantization = True

eval_batch_size = value_batch_size
num_calibration_batches = math.ceil(X_train.size(0)/eval_batch_size)
num_eval_batches = math.ceil(X_train.size(0)/eval_batch_size)



model_quantized.eval()



## Fuse Conv, bn and relu
#model_quantized.fuse_model()



# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model_quantized.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.backends.quantized.engine = 'qnnpack'
print(model_quantized.qconfig)

torch.quantization.prepare(model_quantized, inplace=True)

print(f'\n{color.NONE}Post-training quantization{color.END}')

# Calibrate first
# print('Prepare: Inserting Observers')
#print('\n Inverted Residual Block:After observer insertion \n\n', model_quantized.features[1].conv)

# Calibrate with the training set
evaluate(model_quantized, criterion, loader_train, neval_batches=num_calibration_batches)
# print('Post Training Quantization: Calibration done')

# Convert to quantized model
t_done = False
t_dict = {'prefix' : 'Covert: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()
torch.quantization.convert(model_quantized, inplace=True)
t_done = True
time.sleep(0.2)
# print('Post Training Quantization: Convert done')

# top1, top5 = evaluate(model_quantized, criterion, loader_valid, neval_batches=num_eval_batches)
# print('\n\nEvaluation accuracy on %d samples, %.3f'%(num_eval_batches * eval_batch_size, top1.avg))

save_model(model_quantized, '_quantized')









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗     █████╗ ██╗    ██╗ █████╗ ██████╗ ███████╗ ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔════╝ ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ███████║██║ █╗ ██║███████║██████╔╝█████╗█████╗██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══██║██║███╗██║██╔══██║██╔══██╗██╔══╝╚════╝██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
# ╚═══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

#model_temp = model

#num_trainepoch_quant = 8
#optimizer = torch.optim.SGD(model_temp.parameters(), lr = 0.005)
#eval_batch_size = value_batch_size
#num_train_batches = math.ceil(round(x.size(0)*value_dataset_split,0)/eval_batch_size)
#num_eval_batches = math.ceil(round(x.size(0)*(1-value_dataset_split),0)/eval_batch_size)

#model_temp.fuse_model()
#model_temp.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
#torch.quantization.prepare_qat(model_temp, inplace=True)




## Train and check accuracy after each epoch
#for nepoch in range(num_trainepoch_quant):
#    train_one_epoch(model_temp, criterion, optimizer, loader_train, torch.device('cpu'), num_train_batches)
#    if nepoch > 3:
#        # Freeze quantizer parameters
#        model_temp.apply(torch.quantization.disable_observer)
#    if nepoch > 2:
#        # Freeze batch norm mean and variance estimates
#        model_temp.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

#    # Check the accuracy after each epoch
#    model_quantized = torch.quantization.convert(model_temp.eval(), inplace=False)
#    model_quantized.eval()
#    top1, top5 = evaluate(model_quantized,criterion, loader_valid, neval_batches=num_eval_batches)
#    print('\nEpoch %d :Evaluation accuracy on %d images, %2.2f\n'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))









#███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#█████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

cnt = 0
cnt_t = 0

cm_cnt = 0
cmatrix_temp = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)
# accmatrix = np.zeros(fully_2_outdim)
cmatrix = np.zeros((len(data_names), fully_2_outdim, fully_2_outdim), dtype=int)
# cmatrix = [[[[] for j in range(fully_2_outdim)] for k in range(fully_2_outdim)] for d in data_names]

print('\n\n')
printProgressBar(0, len(loader_valid), prefix = f'{color.NONE}Floating model evaluation{color.END}:', suffix = '', length = 40)
for i, data in enumerate(loader_valid):

    inputs, labels = data
    outputs = model(inputs.float())

    list.clear(max_i)
    for o in outputs:
        m=max(o)
        indx=list(o).index(m)
        max_i.append(indx)

    for idx, tgt in zip(max_i, labels):
        cmatrix[R_valid[cm_cnt]][idx][tgt] += 1
        cm_cnt += 1


    # for o, m in zip(labels, max_i):
    #     if o == m:
    #         cnt = cnt + 1
    #     cnt_t = cnt_t + 1

    printProgressBar(i + 1, len(loader_valid), prefix = f'{color.NONE}Floating model evaluation{color.END}:', suffix = '', length = 40)

# print('\nAccuracy on validation set with floating point model: %f' % (cnt/cnt_t))
#
# cnt = 0
# cnt_t = 0

for matrix in cmatrix:
    cmatrix_temp += matrix
    for i in range(fully_2_outdim):
        cnt += matrix[i][i]
        cnt_t += matrix.sum(axis=0)[i]

print('\nAccuracy on validation set with floating point model: %f' % (cnt/cnt_t))

print('\nConfusion matrix:')
print(cmatrix_temp)



cnt_q = 0
cnt_t_q = 0

cm_cnt = 0
cmatrix_temp = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix_q = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)
# accmatrix_q = np.zeros(fully_2_outdim)
cmatrix_q = np.zeros((len(data_names), fully_2_outdim, fully_2_outdim), dtype=int)
# cmatrix_q = [[[[] for j in range(fully_2_outdim)] for k in range(fully_2_outdim)] for d in data_names]

print('\n\n')
printProgressBar(0, len(loader_valid), prefix = f'{color.NONE}Fixed model evaluation{color.END}:', suffix = '', length = 40)
for i, data in enumerate(loader_valid):

    inputs, labels = data

    outputs = model_quantized(inputs.float())

    list.clear(max_i)
    for o in outputs:
        m=max(o)
        indx=list(o).index(m)
        max_i.append(indx)

    for idx, tgt in zip(max_i, labels):
        cmatrix_q[R_valid[cm_cnt]][idx][tgt] += 1
        cm_cnt += 1


    # for o, m in zip(labels, max_i):
    #     if o == m:
    #         cnt_q = cnt_q + 1
    #     cnt_t_q = cnt_t_q + 1

    printProgressBar(i + 1, len(loader_valid), prefix = f'{color.NONE}Fixed model evaluation{color.END}:', suffix = '', length = 40)

for matrix in cmatrix_q:
    cmatrix_temp += matrix
    for i in range(fully_2_outdim):
        cnt_q += matrix[i][i]
        cnt_t_q += matrix.sum(axis=0)[i]

print('\nAccuracy on validation set with fixed point model: %f' % (cnt_q/cnt_t_q))

# for i in range(fully_2_outdim):
#     accmatrix_q[i] = (cmatrix_q[i][i]/cmatrix_q.sum(axis=0)[i])

print('\nConfusion matrix:')
print(cmatrix_temp)

# print('\nAccuracy per output:')
# print(accmatrix_q)









# ███████╗██╗  ██╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗     ██████╗  █████╗ ████████╗ █████╗
# ██╔════╝╚██╗██╔╝██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝     ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# █████╗   ╚███╔╝ ██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗    ██║  ██║███████║   ██║   ███████║
# ██╔══╝   ██╔██╗ ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║    ██║  ██║██╔══██║   ██║   ██╔══██║
# ███████╗██╔╝ ██╗██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝    ██████╔╝██║  ██║   ██║   ██║  ██║
# ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

print('\n\n')
t_done = False
t_dict = {'prefix' : f'{color.NONE}Exporting data{color.END}: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

model_quantized.debug = True

for i, data in enumerate(loader_valid):

    inputs, labels = data

    if(model_quantized.debug):

        for j in range(0,value_batch_size):
            model_quantized.temp = j
            outputs = model_quantized(inputs[j].unsqueeze_(0).float())

        torch.set_printoptions(threshold=500000, precision=10) #,linehalf_windowth=20
        f = open(session_path+"inference_data_example/labels.txt", "w")
        f.write(str(labels))
        f.close()

    break

training_parameters = {
    'session_name': session_name,
    'dataset_name' : dataset_name,
    'dataset_split' : dataset_split,
    'augmentation' : False if augmentation == [0, 1] else augmentation,
    'random_seed' : random_seed,
    'batch_size' : value_batch_size,
    'normalization' : normalization,
    'conv_indim' : conv_indim,
    'pool_ks' : pool_ks,
    'conv_1_if' : conv_1_if,
    'conv_1_of' : conv_1_of,
    'conv_1_ks' : conv_1_ks,
    'conv_2_if' : conv_2_if,
    'conv_2_of' : conv_2_of,
    'conv_2_ks' : conv_2_ks,
    'fully_1_indim' : fully_1_indim,
    'fully_1_outdim' : fully_1_outdim,
    'fully_2_indim' : fully_2_indim,
    'fully_2_outdim' : fully_2_outdim,
    'train_epoch' : num_trainepoch_effective,
    'optimizer' : str(optimizer).replace("\n","").replace("    ",", "),
    'criterion' : str(criterion),
    'training_acc' : train_data[train_dic['train_acc']][-1],
    'validation_acc' : train_data[train_dic['valid_acc']][-1],
    'training_loss' : train_data[train_dic['train_loss']][-1],
    'validation_loss' : train_data[train_dic['valid_loss']][-1],
    'learning_rate' : train_data[train_dic['learing_rate']][-1],
    'fixed_point_accuracy' : (cnt_q/cnt_t_q)
}

with open(session_path+'training_summary.json', 'w') as json_file:
    json.dump(training_parameters, json_file, indent=4)

# with open(session_path+'training_loss.pickle', 'wb') as output_file:
#     pk.dump(training_loss, output_file)
#
# with open(session_path+'training_acc.pickle', 'wb') as output_file:
#     pk.dump(training_acc, output_file)

with open(session_path+'training_data.pickle', 'wb') as output_file:
    pk.dump(train_data, output_file)

with open(session_path+'confusionmatrix_float.pickle', 'wb') as output_file:
    pk.dump(cmatrix, output_file)

with open(session_path+'confusionmatrix_fixed.pickle', 'wb') as output_file:
    pk.dump(cmatrix_q, output_file)

t_done = True
time.sleep(0.2)


print(f'\n{color.NONE}Summary{color.END}')
for par in training_parameters:
    print(repr(par),":",training_parameters[par])

print(f'{color.BOLD}\n\n\nEnding {color.NONE}training{color.END}{color.BOLD} session \'{session_name}\'{color.END}')
