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
parser.add_argument('-p','--peak', dest='peak', required=True, help="peak detection session path")
parser.add_argument('-t','--train', dest='train', required=True, help="training session path")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")

args = parser.parse_args()



session_name = args.name
session_path = "output/evaluation/"+session_name+"/"
if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
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
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

print(f'{color.BOLD}Starting {color.NONE}evaluation{color.END}{color.BOLD} session \'{session_name}\'\n\n\n{color.END}')

session_train_path = args.train
if os.path.exists(session_train_path) == False:
    print(session_train_path+" does not exist!")
    exit()

session_peak_path = args.peak
if os.path.exists(session_peak_path) == False:
    print(session_peak_path+" does not exist!")
    exit()

json_file = open(session_train_path+'/training_summary.json', 'r')
json_data = json.load(json_file)

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        printProgressBar(0, len(data_loader), prefix = 'Post training:', suffix = '', length = 55)
        for image, target in data_loader:
            output = model(image.float())
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            printProgressBar(cnt, len(data_loader), prefix = 'Post training:', suffix = '', length = 55)
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

t_done = False
def animate(prefix = ''):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if t_done:
            break
        print('\r' + prefix + c, end = '\r')
        # sys.stdout.write('\r' + prefix + c)
        # sys.stdout.flush()
        time.sleep(0.1)
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
        print()

criterion = nn.CrossEntropyLoss()









#██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
#██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
#██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
#██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
#██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
#╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝

dataset_name = json_data['dataset_name']
dataset_split = float(json_data['dataset_split'])
value_dataset_split = json_data['dataset_split']
value_batch_size = json_data['batch_size']
normalization = json_data['normalization']
augmentation = json_data['augmentation'] if json_data['augmentation'] else [0, 1]
random_seed = json_data['random_seed']
half_window = 99



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



# if 'U' in labels:
#     peak_path = args.peak
#     with open(f'{peak_path}/matrix_l.pickle', 'rb') as input_file:
#         matrix_l = pk.load(input_file)

printProgressBar(0, len(data_names), prefix = 'Dataset building:', suffix = '', length = 50)
for d in data_names:
    r = wfdb.rdrecord('./dataset/raw/'+d)
    ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    sig = np.array(r.p_signal[:,0])
    intsig = np.array(r.p_signal[:,0])
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample
    if 'U' in labels:
        sym_len = len(sym)
        for i in range(0, sym_len, 4):
            if int(pos[i] + pos[i+1]) > 100:
                sym.append('U')
                pos = np.append(pos,[int((pos[i] + pos[i+1]) / 2)])
    # if 'U' in labels:
    #     print(d)
    #     for matrix, i in zip(matrix_l[data_names.index(d)], range(len(matrix_l[data_names.index(d)]))):
    #         if len(matrix) == 0:
    #             sym.append('U')
    #             pos = np.append(pos,pos[i])
    beat_len = len(sym)
    for i in range(beat_len):
        for j in range(-augmentation[0]*augmentation[1],augmentation[0]*augmentation[1]+1,augmentation[1]):
            if pos[i]-half_window+j>=0 and pos[i]+half_window+j<=sig_len and sym[i] in sub_labels:
                frame = sig[pos[i]-half_window+j:pos[i]+half_window+j]
                X.append(frame)
                Y.append(labels.index(sub_labels[sym[i]]))
                C.append(True if j == 0 else False)
                R.append(data_names.index(d))
                P.append(pos[i])
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
P = np.array(P)[item_perm]

X_train = X[:round(np.size(X,0)*dataset_split)]
Y_train = Y[:round(np.size(X,0)*dataset_split)]
# C_train = C[:round(np.size(X,0)*dataset_split)]
# R_train = R[:round(np.size(X,0)*dataset_split)]
# P_train = P[:round(np.size(X,0)*dataset_split)]

X_valid = X[round(np.size(X,0)*dataset_split):]
Y_valid = Y[round(np.size(X,0)*dataset_split):]
C_valid = C[round(np.size(X,0)*dataset_split):]
R_valid = R[round(np.size(X,0)*dataset_split):]
P_valid = P[round(np.size(X,0)*dataset_split):]

X_valid = X_valid[C_valid]
Y_valid = Y_valid[C_valid]
R_valid = R_valid[C_valid]
P_valid = P_valid[C_valid]






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

loader_train = torch.utils.data.DataLoader(t_dataset_train, batch_size=value_batch_size, shuffle=True)
loader_valid = torch.utils.data.DataLoader(t_dataset_valid, batch_size=value_batch_size, shuffle=True)

t_done = True
time.sleep(0.1)
print('\n')








#███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
#████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
#██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
#██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
#██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
#╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

conv_indim = json_data['conv_indim']

pool_ks = json_data['pool_ks']

conv_1_if = json_data['conv_1_if']
conv_1_of = json_data['conv_1_of']
conv_1_ks = json_data['conv_1_ks']

conv_2_if = json_data['conv_2_if']
conv_2_of = json_data['conv_2_of']
conv_2_ks = json_data['conv_2_ks']

fully_1_indim = json_data['fully_1_indim']
fully_1_outdim = json_data['fully_1_outdim']

fully_2_indim = json_data['fully_2_indim']
fully_2_outdim = json_data['fully_2_outdim']



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
            torch.set_printoptions(threshold=500000, precision=10) #, linewidth=20
            f = open(session_train_path+"/inference_data_example/input_"+str(self.temp)+".txt", "w")

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









# ██╗      ██████╗  █████╗ ██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
# ██║     ██╔═══██╗██╔══██╗██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
# ██║     ██║   ██║███████║██║  ██║    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
# ██║     ██║   ██║██╔══██║██║  ██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
# ███████╗╚██████╔╝██║  ██║██████╔╝    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
# ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

model = Net()
model.load_state_dict(torch.load(session_train_path+'/model.pth'))
model.eval()

model_quantized = copy.deepcopy(model)
model_quantized.quantization = True

eval_batch_size = value_batch_size
num_calibration_batches = math.ceil(X_train.size(0)/eval_batch_size)
num_eval_batches = math.ceil(X_train.size(0)/eval_batch_size)



print(f'\n{color.NONE}Post-training quantization{color.END}')
model_quantized.eval()
model_quantized.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.backends.quantized.engine = 'qnnpack'
torch.quantization.prepare(model_quantized, inplace=True)

evaluate(model_quantized, criterion, loader_train, neval_batches=num_calibration_batches)

t_done = False
t_dict = {'prefix' : 'Covert: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()
torch.quantization.convert(model_quantized, inplace=True)
t_done = True
time.sleep(0.1)









# ██████╗ ███████╗ █████╗ ██╗  ██╗    ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔══██╗██╔════╝██╔══██╗██║ ██╔╝    ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██████╔╝█████╗  ███████║█████╔╝     █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗     ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ██║     ███████╗██║  ██║██║  ██╗    ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

wid = 99



with open(session_peak_path+'/peakpos.pickle', "rb") as output_file:
    PEAKPOS = pk.load(output_file)

with open(f'{session_peak_path}/matrix_l.pickle', 'rb') as input_file:
    matrix_l = pk.load(input_file)

with open(f'{session_peak_path}/matrix_p.pickle', 'rb') as input_file:
    matrix_p = pk.load(input_file)

with open(f'{session_peak_path}/fp_pos.pickle', 'rb') as input_file:
    fp_pos = pk.load(input_file)



max_i = []

valid_filter = []
X_valid_filtered = []
Y_valid_filtered = []
P_valid_filtered = []

tp_cmatrix_float = []
tp_cmatrix_float_centered = []
tp_cmatrix_fixed = []
tp_cmatrix_fixed_centered = []
tp_cmatrix = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

fp_vector = []

# printProgressBar(0, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 55)
for d in data_names:
    tp_cmatrix_float.append([[(0) for i in range(len(labels))] for i in range(len(labels))])
    tp_cmatrix_float_centered.append([[(0) for i in range(len(labels))] for i in range(len(labels))])
    tp_cmatrix_fixed.append([[(0) for i in range(len(labels))] for i in range(len(labels))])
    tp_cmatrix_fixed_centered.append([[(0) for i in range(len(labels))] for i in range(len(labels))])

    fp_vector.append([])

    r = wfdb.rdrecord('./dataset/raw/'+d)
    ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    sig = np.array(r.p_signal[:,0])
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample

    list.clear(valid_filter)
    for r in R_valid:
        if r == data_names.index(d):
            valid_filter.append(True)
        else:
            valid_filter.append(False)
    P_valid_filtered = P_valid[valid_filter]
    Y_valid_filtered = Y_valid[valid_filter]

    for couple in matrix_l[data_names.index(d)]:
        if len(couple):
            if couple[0] in P_valid_filtered:

                if (couple[1] - wid >= 0) and (couple[1] + wid < sig_len):
                    input =  torch.from_numpy(sig[couple[1]-wid:couple[1]+wid])
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)

                    output = model(input.float())
                    m=max(output[0])
                    indx=list(output[0]).index(m)

                    tp_cmatrix_float[data_names.index(d)][int(Y_valid_filtered[np.where(P_valid_filtered == couple[0])[0][0]])][indx] += 1

                if (couple[0] - wid >= 0) and (couple[0] + wid < sig_len):
                    input =  torch.from_numpy(sig[couple[0]-wid:couple[0]+wid])
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)

                    output = model(input.float())
                    m=max(output[0])
                    indx=list(output[0]).index(m)

                    tp_cmatrix_float_centered[data_names.index(d)][int(Y_valid_filtered[np.where(P_valid_filtered == couple[0])[0][0]])][indx] += 1

                if (couple[1] - wid >= 0) and (couple[1] + wid < sig_len):
                    input =  torch.from_numpy(sig[couple[1]-wid:couple[1]+wid])
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)

                    output = model_quantized(input.float())
                    m=max(output[0])
                    indx=list(output[0]).index(m)

                    tp_cmatrix_fixed[data_names.index(d)][int(Y_valid_filtered[np.where(P_valid_filtered == couple[0])[0][0]])][indx] += 1

                if (couple[0] - wid >= 0) and (couple[0] + wid < sig_len):
                    input =  torch.from_numpy(sig[couple[0]-wid:couple[0]+wid])
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)
                    input.unsqueeze_(0)

                    output = model_quantized(input.float())
                    m=max(output[0])
                    indx=list(output[0]).index(m)

                    tp_cmatrix_fixed_centered[data_names.index(d)][int(Y_valid_filtered[np.where(P_valid_filtered == couple[0])[0][0]])][indx] += 1

    for f in fp_pos[data_names.index(d)]:
        if (f - wid >= 0) and (f + wid < sig_len):
            input = torch.from_numpy(sig[f-wid:f+wid])
            input.unsqueeze_(0)
            input.unsqueeze_(0)
            input.unsqueeze_(0)

            output = model_quantized(input.float())
            m = max(output[0])
            indx = list(output[0]).index(m)

            fp_vector[data_names.index(d)].append(indx)

    print(f'\n\nfile: {d}')
    d_sum = 0
    i_sum = 0
    j_sum = 0
    for i in range(len(tp_cmatrix_fixed[data_names.index(d)])):
        for j in range(len(tp_cmatrix_fixed[data_names.index(d)][i])):
            print(f'{tp_cmatrix_fixed[data_names.index(d)][i][j]:04d}    ',end='')
            j_sum += tp_cmatrix_fixed[data_names.index(d)][i][j]
        d_sum += tp_cmatrix_fixed[data_names.index(d)][i][i]
        i_sum += j_sum
        if j_sum:
            print(f'{tp_cmatrix_fixed[data_names.index(d)][i][i]/j_sum}')
        else:
            print('-')
        j_sum = 0
    if i_sum:
        print(f'total accuracy:                         {d_sum/i_sum}\n')
    else:
        print(f'-\n')

    print(f'file: {d} (centered)')
    d_sum = 0
    i_sum = 0
    j_sum = 0
    for i in range(len(tp_cmatrix_fixed_centered[data_names.index(d)])):
        for j in range(len(tp_cmatrix_fixed_centered[data_names.index(d)][i])):
            print(f'{tp_cmatrix_fixed_centered[data_names.index(d)][i][j]:04d}    ',end='')
            j_sum += tp_cmatrix_fixed_centered[data_names.index(d)][i][j]
        d_sum += tp_cmatrix_fixed_centered[data_names.index(d)][i][i]
        i_sum += j_sum
        if j_sum:
            print(f'{tp_cmatrix_fixed_centered[data_names.index(d)][i][i]/j_sum}')
        else:
            print('-')
        j_sum = 0
    if i_sum:
        print(f'total accuracy:                         {d_sum/i_sum}\n')
    else:
        print(f'-\n')



print(fp_vector)
cnt_t = 0
cnt_n = 0

for fp in fp_vector:
    for f in fp:
        if f == 0:
            cnt_n += 1
        cnt_t += 1

print(cnt_t)
print(cnt_n)




print('\n')
t_done = False
t_dict = {'prefix' : f'{color.NONE}Exporting data{color.END}: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

with open(session_path+'tp_cmatrix_fixed.pickle', 'wb') as output_file:
    pk.dump(tp_cmatrix_fixed, output_file)
with open(session_path+'tp_cmatrix_fixed_centered.pickle', 'wb') as output_file:
    pk.dump(tp_cmatrix_fixed_centered, output_file)
with open(session_path+'tp_cmatrix_float.pickle', 'wb') as output_file:
    pk.dump(tp_cmatrix_float, output_file)
with open(session_path+'tp_cmatrix_float_centered.pickle', 'wb') as output_file:
    pk.dump(tp_cmatrix_float_centered, output_file)

training_parameters = {
    'session_name': session_name,
    'session_train_path' : session_train_path,
    'session_peak_path' : session_peak_path
}

with open(session_path+'evaluation_summary.json', 'w') as json_file:
    json.dump(training_parameters, json_file, indent=2)

t_done = True
time.sleep(0.2)

print(f'{color.BOLD}\n\n\nEnding {color.NONE}evaluation{color.END}{color.BOLD} session \'{session_name}\'{color.END}')

matrix = np.zeros((5,5))
for cmat in tp_cmatrix_fixed_centered:
    matrix = np.add(matrix,np.array(cmat))
np.set_printoptions(suppress=True),
print(matrix)
