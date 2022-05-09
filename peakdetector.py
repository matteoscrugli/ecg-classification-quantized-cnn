import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys, os, time
import argparse
from pathlib import Path
import shutil
import pickle as pk
import wfdb
import pywt
from collections import Counter
from PIL import Image
from PIL import ImageOps

import itertools
import threading

parser = argparse.ArgumentParser()

parser.add_argument('-n','--name', dest='name', required=True, help="session name")
parser.add_argument('-t','--tolerance', dest='tolerance', required=True, type=int, help="spatial tolerance of peak recognition compared to the real one")
parser.add_argument('-m','--multiplier', dest='multiplier', default='1', help="threshold multiplier")
parser.add_argument('-norm','--normalize', dest='normalize', action='store_true', help="normalize bar plot")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")

args = parser.parse_args()



session_name = args.name
session_path = "output/peakdetector/"+session_name+"/"
if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in session creation ("+session_path+").")
            exit()
    else:
        print("Session already exists ("+session_path+"), overwrite the session? (y/n): ", end='')
        force_write = input()
        if force_write == "y":
            print('')
            try:
                shutil.rmtree(session_path)
                Path(session_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                print("Error in session creation ("+session_path+").")
                exit()
        else:
            exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()



class color:
    NONE = ''
    GRAY = '\033[90m'
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
    # Print New p on Complete
    if iteration == total:
        print(f'\r{prefix} |{bar}| Done! {suffix}')

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



#
# data_names = ['100', '101', '102', '103', '104', '105', '106', '107',
#               '108', '109', '111', '112', '113', '114', '115', '116',
#               '117', '118', '119', '121', '122', '123', '124', '200',
#               '201', '202', '203', '205', '207', '208', '209', '210',
#               '212', '213', '214', '215', '217', '219', '220', '221',
#               '222', '223', '228', '230', '231', '232', '233', '234']

data_names = ['100', '101', '103', '105', '106', '107',
              '108', '109', '111', '112', '113', '114', '115', '116',
              '117', '118', '119', '121', '122', '123', '124', '200',
              '201', '202', '203', '205', '207', '208', '209', '210',
              '212', '213', '214', '215', '217', '219', '220', '221',
              '222', '223', '228', '230', '231', '232', '233', '234']




# data_names = ['100', '101', '102', '103']

# float2int_mul = 10000;
wid = 99

# X = []
# Y = []
# F = []
# P = []
#
# X_NLRAV = []
# Y_NLRAV = []
# F_NLRAV = []
# P_NLRAV = []
#
# X_NSVFQ = []
# Y_NSVFQ = []
# F_NSVFQ = []
# P_NSVFQ = []
#
# labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
# NLRAV_labels = ['N', 'L', 'R', 'A', 'V']
# NSVFQ_labels = ['N', 'S', 'V', 'F', 'Q']
# NSVFQ_sub_labels = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
#        'A':'S', 'a':'S', 'J':'S', 'S':'S',
#        'V':'V', 'E':'V',
#        'F':'F',
#        '/':'Q', 'f':'Q', 'Q':'Q'}
# X = []
# Y = []
# allstat_min = 0
# allstat_max = 0
#
# temp_path = './output/dataset/raw_text/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     while os.path.exists(temp_path): # check if it exists
#         pass
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#     while not os.path.exists(temp_path): # check if it exists
#         pass
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# time.sleep(6)
#
# f_allstat = open('./output/dataset/raw_text/all_stat.txt', "w")
#
# printProgressBar(0, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
# for d in data_names:
#     f_data = open('./output/dataset/raw_text/'+d+'_data.txt', "w")
#     f_intdata = open('./output/dataset/raw_text/'+d+'_intdata.txt', "w")
#     f_labels = open('./output/dataset/raw_text/'+d+'_label.txt', "w")
#     f_labelspos = open('./output/dataset/raw_text/'+d+'_labelpos.txt', "w")
#     f_labels_NLRAV = open('./output/dataset/raw_text/'+d+'_label_NLRAV.txt', "w")
#     f_labelspos_NLRAV = open('./output/dataset/raw_text/'+d+'_labelpos_NLRAV.txt', "w")
#     f_labels_NSVFQ = open('./output/dataset/raw_text/'+d+'_label_NSVFQ.txt', "w")
#     f_labelspos_NSVFQ = open('./output/dataset/raw_text/'+d+'_labelpos_NSVFQ.txt', "w")
#     f_stat = open('./output/dataset/raw_text/'+d+'_stat.txt', "w")
#     r = wfdb.rdrecord('./dataset/raw/'+d)
#     ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
#     sig = np.array(r.p_signal[:,0])
#     intsig = np.array(r.p_signal[:,0])
#     sig_len = len(sig)
#     sym = ann.symbol
#     pos = ann.sample
#     beat_len = len(sym)
#     stat_min = sig[0]
#     stat_max = sig[0]
#     for i in range(sig_len):
#         f_data.write(str(sig[i])+'\n')
#         intsig[i]=int(sig[i]*float2int_mul)
#         f_intdata.write(str(intsig[i])+'\n')
#         if sig[i] < stat_min:
#             stat_min = sig[i]
#         if sig[i] > stat_max:
#             stat_max = sig[i]
#     for i in range(beat_len):
#         if pos[i]-wid>=0 and pos[i]+wid<=sig_len and sym[i] in labels:
#             f_labels.write(str(sym[i])+'\n')
#             f_labelspos.write(str(pos[i])+'\n')
#
#             a = sig[pos[i]-wid:pos[i]+wid]
#             if len(a) != 2*wid:
#                 print("Length error")
#                 continue
#             X.append(a)
#             Y.append(labels.index(sym[i]))
#             F.append(data_names.index(d))
#             P.append(pos[i])
#
#             if sym[i] in NLRAV_labels:
#                 f_labels_NLRAV.write(str(sym[i])+'\n')
#                 f_labelspos_NLRAV.write(str(pos[i])+'\n')
#
#                 X_NLRAV.append(a)
#                 Y_NLRAV.append(NLRAV_labels.index(sym[i]))
#                 F_NLRAV.append(data_names.index(d))
#                 P_NLRAV.append(pos[i])
#
#             f_labels_NSVFQ.write(str(NSVFQ_sub_labels[sym[i]])+'\n')
#             f_labelspos_NSVFQ.write(str(pos[i])+'\n')
#
#             X_NSVFQ.append(a)
#             Y_NSVFQ.append(NSVFQ_labels.index(NSVFQ_sub_labels[sym[i]]))
#             F_NSVFQ.append(data_names.index(d))
#             P_NSVFQ.append(pos[i])
#     if d == data_names[0]:
#         allstat_min = stat_min
#         allstat_max = stat_max
#     else:
#         if stat_min < allstat_min:
#             allstat_min = stat_min
#         if stat_max > allstat_max:
#             allstat_max = stat_max
#
#     f_stat.write(str(stat_min)+'\n'+str(stat_max))
#     f_data.close()
#     f_intdata.close()
#     f_labels.close()
#     f_labelspos.close()
#     f_labels_NLRAV.close()
#     f_labelspos_NLRAV.close()
#     f_labels_NSVFQ.close()
#     f_labelspos_NSVFQ.close()
#     f_stat.close()
#     printProgressBar(data_names.index(d) + 1, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
#
# f_allstat.write(str(allstat_min)+'\n'+str(allstat_max))
# f_allstat.close()
#
#
#
# t_done = False
# t_dict = {'prefix' : 'Exporting dataset peaks: '}
# t = threading.Thread(target=animate, kwargs=t_dict)
# t.start()
#
# X = np.array(X)
# Y = np.array(Y)
# F = np.array(F)
# P = np.array(P)
#
# X_NLRAV = np.array(X_NLRAV)
# Y_NLRAV = np.array(Y_NLRAV)
# F_NLRAV = np.array(F_NLRAV)
# P_NLRAV = np.array(P_NLRAV)
#
# X_NSVFQ = np.array(X_NSVFQ)
# Y_NSVFQ = np.array(Y_NSVFQ)
# F_NSVFQ = np.array(F_NSVFQ)
# P_NSVFQ = np.array(P_NSVFQ)
#
# temp_path = './output/dataset/raw_peaks/ordered/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks/ordered/dataset_data.pickle", "wb") as output_file:
#     pk.dump(X, output_file)
# with open("./output/dataset/raw_peaks/ordered/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y, output_file)
# with open("./output/dataset/raw_peaks/ordered/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F, output_file)
# with open("./output/dataset/raw_peaks/ordered/dataset_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P, output_file)
#
#
#
# temp_path = './output/dataset/raw_peaks/ordered_NLRAV/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_NLRAV_data.pickle", "wb") as output_file:
#     pk.dump(X_NLRAV, output_file)
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_NLRAV_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NLRAV, output_file)
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_NLRAV_file.pickle", "wb") as output_file:
#     pk.dump(F_NLRAV, output_file)
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_NLRAV_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NLRAV, output_file)
#
#
#
# temp_path = './output/dataset/raw_peaks/ordered_NSVFQ/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_NSVFQ_data.pickle", "wb") as output_file:
#     pk.dump(X_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_NSVFQ_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_NSVFQ_file.pickle", "wb") as output_file:
#     pk.dump(F_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_NSVFQ_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NSVFQ, output_file)
#
# t_done = True
# time.sleep(0.1)
#
#
#
#
# X = []
# Y = []
# F = []
# P = []
#
# X_NLRAV = []
# Y_NLRAV = []
# F_NLRAV = []
# P_NLRAV = []
#
# X_NSVFQ = []
# Y_NSVFQ = []
# F_NSVFQ = []
# P_NSVFQ = []
#
# labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
# NLRAV_labels = ['N', 'L', 'R', 'A', 'V']
# NSVFQ_labels = ['N', 'S', 'V', 'F', 'Q']
# NSVFQ_sub_labels = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
#        'A':'S', 'a':'S', 'J':'S', 'S':'S',
#        'V':'V', 'E':'V',
#        'F':'F',
#        '/':'Q', 'f':'Q', 'Q':'Q'}
# X = []
# Y = []
# allstat_min = 0
# allstat_max = 0
#
# temp_path = './output/dataset/raw_text_aug/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     while os.path.exists(temp_path): # check if it exists
#         pass
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#     while not os.path.exists(temp_path): # check if it exists
#         pass
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# time.sleep(6)
#
#
#
# f_allstat = open('./output/dataset/raw_text_aug/all_stat.txt', "w")
#
# printProgressBar(0, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
# for d in data_names:
#     f_data = open('./output/dataset/raw_text_aug/'+d+'_data.txt', "w")
#     f_intdata = open('./output/dataset/raw_text_aug/'+d+'_intdata.txt', "w")
#     f_labels = open('./output/dataset/raw_text_aug/'+d+'_label.txt', "w")
#     f_labelspos = open('./output/dataset/raw_text_aug/'+d+'_labelpos.txt', "w")
#     f_labels_NLRAV = open('./output/dataset/raw_text_aug/'+d+'_label_NLRAV.txt', "w")
#     f_labelspos_NLRAV = open('./output/dataset/raw_text_aug/'+d+'_labelpos_NLRAV.txt', "w")
#     f_labels_NSVFQ = open('./output/dataset/raw_text_aug/'+d+'_label_NSVFQ.txt', "w")
#     f_labelspos_NSVFQ = open('./output/dataset/raw_text_aug/'+d+'_labelpos_NSVFQ.txt', "w")
#     f_stat = open('./output/dataset/raw_text_aug/'+d+'_stat.txt', "w")
#     r = wfdb.rdrecord('./dataset/raw/'+d)
#     ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
#     sig = np.array(r.p_signal[:,0])
#     intsig = np.array(r.p_signal[:,0])
#     sig_len = len(sig)
#     sym = ann.symbol
#     pos = ann.sample
#     beat_len = len(sym)
#     stat_min = sig[0]
#     stat_max = sig[0]
#     for i in range(sig_len):
#         f_data.write(str(sig[i])+'\n')
#         intsig[i]=int(sig[i]*float2int_mul)
#         f_intdata.write(str(intsig[i])+'\n')
#         if sig[i] < stat_min:
#             stat_min = sig[i]
#         if sig[i] > stat_max:
#             stat_max = sig[i]
#     for i in range(beat_len):
#         for j in range(-20,21,4):
#             if pos[i]-wid+j>=0 and pos[i]+wid+j<=sig_len and sym[i] in labels:
#                 f_labels.write(str(sym[i])+'\n')
#                 f_labelspos.write(str(pos[i]+j)+'\n')
#
#                 a = sig[pos[i]-wid+j:pos[i]+wid+j]
#                 if len(a) != 2*wid:
#                     print("Length error")
#                     continue
#                 X.append(a)
#                 Y.append(labels.index(sym[i]))
#                 F.append(data_names.index(d))
#                 P.append(pos[i]+j)
#
#                 if sym[i] in NLRAV_labels:
#                     f_labels_NLRAV.write(str(sym[i])+'\n')
#                     f_labelspos_NLRAV.write(str(pos[i]+j)+'\n')
#
#                     X_NLRAV.append(a)
#                     Y_NLRAV.append(NLRAV_labels.index(sym[i]))
#                     F_NLRAV.append(data_names.index(d))
#                     P_NLRAV.append(pos[i]+j)
#
#                 f_labels_NSVFQ.write(str(NSVFQ_sub_labels[sym[i]])+'\n')
#                 # f_labels_NSVFQ.write(f'{NSVFQ_sub_labels[sym[i]]}\n')
#                 f_labelspos_NSVFQ.write(str(pos[i]+j)+'\n')
#
#                 X_NSVFQ.append(a)
#                 Y_NSVFQ.append(NSVFQ_labels.index(NSVFQ_sub_labels[sym[i]]))
#                 F_NSVFQ.append(data_names.index(d))
#                 P_NSVFQ.append(pos[i]+j)
#     if d == data_names[0]:
#         allstat_min = stat_min
#         allstat_max = stat_max
#     else:
#         if stat_min < allstat_min:
#             allstat_min = stat_min
#         if stat_max > allstat_max:
#             allstat_max = stat_max
#
#     f_stat.write(str(stat_min)+'\n'+str(stat_max))
#     f_data.close()
#     f_intdata.close()
#     f_labels.close()
#     f_labelspos.close()
#     f_labels_NLRAV.close()
#     f_labelspos_NLRAV.close()
#     f_labels_NSVFQ.close()
#     f_labelspos_NSVFQ.close()
#     f_stat.close()
#     printProgressBar(data_names.index(d) + 1, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
#
# f_allstat.write(str(allstat_min)+'\n'+str(allstat_max))
# f_allstat.close()
#
#
#
# t_done = False
# t_dict = {'prefix' : 'Exporting dataset peaks: '}
# t = threading.Thread(target=animate, kwargs=t_dict)
# t.start()
#
# X = np.array(X)
# Y = np.array(Y)
# F = np.array(F)
# P = np.array(P)
#
# X_NLRAV = np.array(X_NLRAV)
# Y_NLRAV = np.array(Y_NLRAV)
# F_NLRAV = np.array(F_NLRAV)
# P_NLRAV = np.array(P_NLRAV)
#
# X_NSVFQ = np.array(X_NSVFQ)
# Y_NSVFQ = np.array(Y_NSVFQ)
# F_NSVFQ = np.array(F_NSVFQ)
# P_NSVFQ = np.array(P_NSVFQ)
#
# temp_path = './output/dataset/raw_peaks_aug/ordered/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks_aug/ordered/dataset_data.pickle", "wb") as output_file:
#     pk.dump(X, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered/dataset_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P, output_file)
#
#
#
# temp_path = './output/dataset/raw_peaks_aug/ordered_NLRAV/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_NLRAV_data.pickle", "wb") as output_file:
#     pk.dump(X_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_NLRAV_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_NLRAV_file.pickle", "wb") as output_file:
#     pk.dump(F_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_NLRAV_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NLRAV, output_file)
#
#
#
# temp_path = './output/dataset/raw_peaks_aug/ordered_NSVFQ/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_NSVFQ_data.pickle", "wb") as output_file:
#     pk.dump(X_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_NSVFQ_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_NSVFQ_file.pickle", "wb") as output_file:
#     pk.dump(F_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_NSVFQ_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NSVFQ, output_file)
#
# t_done = True
# time.sleep(0.1)
#
# print('end')
# exit()




t_done = False
t_dict = {'prefix' : 'Signal peaks detection: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

th_multiplier = args.multiplier
os.system(f'python3 tool_threshold.py -n default -m {th_multiplier} -o')
os.system('./peakdetector')

t_done = True
time.sleep(0.1)



PEAKPOS = [[] for i in range(len(data_names))]
for d in data_names:
    f_pos = open('./output/dataset/raw_text/'+d+'_peakpos.txt', 'r')
    Lines = f_pos.readlines()
    for line in Lines:
        PEAKPOS[data_names.index(d)].append(int(line.strip()))
    f_pos.close()
    os.remove('./output/dataset/raw_text/'+d+'_peakpos.txt')

PEAKPOS = np.array(PEAKPOS)
with open(session_path+'peakpos.pickle', "wb") as output_file:
    pk.dump(PEAKPOS, output_file)



# TP = 0
# FP = 0
# FN = 0
# all_TP = 0
# all_FP = 0
# all_FN = 0
labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
peak_label = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
peak_sub_labels = { 'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
                   'A':'S', 'a':'S', 'J':'S', 'S':'S',
                   'V':'V', 'E':'V',
                   'F':'F',
                   '/':'Q', 'f':'Q', 'Q':'Q'}
# peak_label = ['N', 'L', 'R', 'A', 'V']



peak_stat = {}
peak_stat_fn = {}
matrix_l = []
matrix_p = []
fp_pos = []
fn_pos = []

ind_file = 0
ind_label = 0
ind_peak = 0
ind_peak_t = 0
tmp_index = 0

pos_filtered = []

tolerance = args.tolerance

printProgressBar(0, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
for d, peakpos in zip(data_names, PEAKPOS):
    matrix_l.append([])
    matrix_p.append([])
    fp_pos.append([])
    fn_pos.append([])
    r = wfdb.rdrecord('./dataset/raw/'+d)
    ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    if d!='114':
        sig = np.array(r.p_signal[:,0])
        intsig = np.array(r.p_signal[:,0])
    else:
        sig = np.array(r.p_signal[:,1])
        intsig = np.array(r.p_signal[:,1])
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample

    # pos_filtered.clear()

    # for s, p in zip(sym, pos):
    #     if s in labels:
    #         pos_filtered.append(p)

    pos_filtered = [p for p, s in zip(pos, sym) if s in labels]
    label_filtered = [s for p, s in zip(pos, sym) if s in labels]

    for [f1, f2] in zip(pos_filtered, label_filtered):
        if f2 in peak_label:
            if peak_sub_labels[f2] not in peak_stat:
                peak_stat[peak_sub_labels[f2]] = 1
            else:
                peak_stat[peak_sub_labels[f2]] += 1



    beat_len = len(sym)
    beat_filtered_len = len(pos_filtered)
    # stat_min = sig[0]
    # stat_max = sig[0]
    find = 0

    # for i in range(beat_len):
    #     matrix_l[ind_file].append([])
    #     for p in peakpos:
    #         if abs(pos[i] - int(p.strip())) <= tolerance:
    #             matrix_l[ind_file][ind_label].append([])
    #             matrix_l[ind_file][ind_label][ind_peak] = pos[i] - int(p.strip())
    #             ind_peak = ind_peak + 1
    #     ind_peak = 0
    #     ind_label = ind_label + 1
    # ind_label = 0
    #
    # for p in peakpos:
    #     matrix_p[ind_file].append([])
    #     for i in range(beat_len):
    #         if abs(pos[i] - int(p.strip())) <= tolerance:
    #             matrix_p[ind_file][ind_peak].append([])
    #             matrix_p[ind_file][ind_peak][ind_label] = pos[i] - int(p.strip())
    #             ind_label = ind_label + 1
    #     ind_label = 0
    #     ind_peak = ind_peak + 1
    # ind_peak = 0

    # tmp_index = 0
    # for i in range(beat_len):
    #     matrix_l[ind_file].append([])
    #     for p in peakpos[tmp_index:]:
    #         if abs(pos[i] - int(p.strip())) <= tolerance:
    #             matrix_l[ind_file][ind_label].append([])
    #             matrix_l[ind_file][ind_label][ind_peak] = pos[i] - int(p.strip())
    #             ind_peak = ind_peak + 1
    #             tmp_index = tmp_index + 1
    #             break
    #     ind_peak = 0
    #     ind_label = ind_label + 1
    # ind_label = 0
    #
    # tmp_index = 0
    # for p in peakpos:
    #     matrix_p[ind_file].append([])
    #     for i in range(beat_len)[tmp_index:]:
    #         if abs(pos[i] - int(p.strip())) <= tolerance:
    #             matrix_p[ind_file][ind_peak].append([])
    #             matrix_p[ind_file][ind_peak][ind_label] = pos[i] - int(p.strip())
    #             ind_label = ind_label + 1
    #             tmp_index = tmp_index + 1
    #             break
    #     ind_label = 0
    #     ind_peak = ind_peak + 1
    # ind_peak = 0










    # tmp_index = 0
    # for i in range(beat_len):
    #     matrix_l[ind_file].append([])
    #     for p in peakpos[tmp_index:]:
    #         if abs(pos[i] - p) <= tolerance:
    #             matrix_l[ind_file][ind_label].append(pos[i])
    #             matrix_l[ind_file][ind_label].append(p)
    #             ind_peak = ind_peak + 1
    #             tmp_index = tmp_index + 1
    #             break
    #         ind_peak_t = ind_peak_t + 1
    #     ind_peak = 0
    #     ind_peak_t = 0
    #     ind_label = ind_label + 1
    # ind_label = 0
    #
    # tmp_index = 0
    # for p in peakpos:
    #     matrix_p[ind_file].append([])
    #     for i in range(beat_len)[tmp_index:]:
    #         if abs(pos[i] - p) <= tolerance:
    #             matrix_p[ind_file][ind_peak].append(pos[i])
    #             matrix_p[ind_file][ind_peak].append(p)
    #             ind_label = ind_label + 1
    #             tmp_index = tmp_index + 1
    #             break
    #     ind_label = 0
    #     ind_peak = ind_peak + 1
    # ind_peak = 0





    tmp_index = 0
    for i in range(beat_filtered_len):
        matrix_l[ind_file].append([])
        for p in peakpos[tmp_index:]:
            tmp_find = False
            if abs(pos_filtered[i] - p) <= tolerance:
                matrix_l[ind_file][ind_label].append(pos_filtered[i])
                matrix_l[ind_file][ind_label].append(p)
                ind_peak = ind_peak + 1
                tmp_index = tmp_index + 1
                tmp_find = True
                break
            ind_peak_t = ind_peak_t + 1
        if not tmp_find:
            fn_pos[ind_file].append([pos_filtered[i], label_filtered[i]])
        ind_peak = 0
        ind_peak_t = 0
        ind_label = ind_label + 1
    ind_label = 0

    tmp_index = 0
    for p in peakpos:
        tmp_find = False
        matrix_p[ind_file].append([])
        for i in range(beat_filtered_len)[tmp_index:]:
            if abs(pos_filtered[i] - p) <= tolerance:
                matrix_p[ind_file][ind_peak].append(pos_filtered[i])
                matrix_p[ind_file][ind_peak].append(p)
                ind_label = ind_label + 1
                tmp_index = tmp_index + 1
                tmp_find = True
                break
        if not tmp_find:
            fp_pos[ind_file].append(p)
        ind_label = 0
        ind_peak = ind_peak + 1
    ind_peak = 0






    ind_file = ind_file + 1

    printProgressBar(data_names.index(d) + 1, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)

ind_file = 0



print(fp_pos)

sum = 0
for f in fp_pos:
    sum += len(f)
print(sum)



print(fn_pos)

sum = 0
for f in fn_pos:
    sum += len(f)
print(sum)

for fn in fn_pos:
    for [f1, f2] in fn:
        if f2 in peak_label:
            if peak_sub_labels[f2] not in peak_stat_fn:
                peak_stat_fn[peak_sub_labels[f2]] = 1
            else:
                peak_stat_fn[peak_sub_labels[f2]] += 1


print('')
print(peak_stat)
print(peak_stat_fn)

peak_stat_sum = 0
for p in peak_stat:
    peak_stat_sum += peak_stat[p]

peak_stat_fn_sum = 0
for p in peak_stat_fn:
    peak_stat_fn_sum += peak_stat_fn[p]


for p in peak_stat:
    peak_stat[p] = peak_stat[p]/peak_stat_sum

for p in peak_stat_fn:
    peak_stat_fn[p] = peak_stat_fn[p]/peak_stat_fn_sum

print('')
print(peak_stat)
print(peak_stat_fn)

print('')






with open(session_path+'matrix_l.pickle', 'wb') as output_file:
    pk.dump(matrix_l, output_file)
with open(session_path+'matrix_p.pickle', 'wb') as output_file:
    pk.dump(matrix_p, output_file)
with open(session_path+'fp_pos.pickle', 'wb') as output_file:
    pk.dump(fp_pos, output_file)



VP = {}
FP = {}
FN = {}

for sub_matrix_l, sub_matrix_p, file in zip(matrix_l, matrix_p, data_names):
    VP[file] = 0
    FP[file] = 0
    FN[file] = 0
    for sub_matrix in sub_matrix_l:
        if len(sub_matrix) == 0:
            FN[file] = FN[file] + 1
    for sub_matrix in sub_matrix_p:
        if len(sub_matrix) == 0:
            FP[file] = FP[file] + 1
        else:
            VP[file] = VP[file] + 1



# d = '207'
# for i, matrix in enumerate(matrix_p[data_names.index(d)]):
#     if len(matrix) == 0:
#             print(pos_filtered[i])
# exit()

bar_len = 50
print('\nFile \tTrue \tFalse \tFalse')
print('\tpos \tpos \tneg\n')

VP_tot = 0
FP_tot = 0
FN_tot = 0
for vp, fp, fn in zip(VP, FP, FN):
    VP_tot += VP[vp]
    FP_tot += FP[fp]
    FN_tot += FN[fn]

    print(f'{color.GRAY}{vp}{color.END} \t{VP[vp]:04d}\t{color.BLUE}{FP[fp]:04d}{color.END}\t{color.RED}{FN[fn]:04d}{color.END} \t', end = '')

    fp_value = round(FP[fp]/(VP[vp] + FP[fp] + FN[fn])*bar_len)
    fn_value = round(FN[fn]/(VP[vp] + FP[fp] + FN[fn])*bar_len)

    # print(f'{color.}', end = '')
    for i in range(bar_len-fp_value-fn_value):
        print('▮', end = '')
    # print(f'{color.END}', end = '')

    print(f'{color.BLUE}', end = '')
    for i in range(fp_value):
        print('▮', end = '')
    print(f'{color.END}', end = '')

    print(f'{color.RED}', end = '')
    for i in range(fn_value):
        print('▮', end = '')
    print(f'{color.END}', end = '')
    print('')

print(f'vp: {VP_tot}    fp: {FP_tot}  fn: {FN_tot}')
# print(f'sensitivity: {VP_tot/(VP_tot + FN_tot)}    precision: {VP_tot/(VP_tot + FP_tot)}')



data_names.reverse()

VP_list = [VP[d] for d in data_names]
FP_list = [FP[d] for d in data_names]
FN_list = [FN[d] for d in data_names]
TOT_list = [VP[d]+FP[d]+FN[d] for d in data_names]

normalize = args.normalize
if normalize:
    VP_list = [vp/TOT_list[i] for i, vp in enumerate(VP_list)]
    FP_list = [fp/TOT_list[i] for i, fp in enumerate(FP_list)]
    FN_list = [fn/TOT_list[i] for i, fn in enumerate(FN_list)]



width = 0.66

# fig, ax = plt.subplots()

plt.figure(figsize=[12, 6])
plt.title(f'Tolerance: {tolerance}')
# plt.suptitle(f"Peaks detected (tolerance of {tolerance} samples)")

p1 = plt.barh(np.arange(len(VP_list)), VP_list, width, alpha= 0.8) #, color='maroon'
p2 = plt.barh(np.arange(len(VP_list)), FP_list, width, left=VP_list, alpha= 0.6) #, color='tab:blue'
p3 = plt.barh(np.arange(len(VP_list)), FN_list, width, left=[vp + fp for vp, fp in zip(VP_list, FP_list)], alpha= 0.6) #, color='tab:green'

plt.ylabel('File')
plt.xlabel('Sample')
plt.yticks(range(len(VP)), data_names) #, rotation=-45
plt.legend((p1[0], p2[0], p3[0]), ('True positive', 'False positive', 'False negative'), loc='lower left')

plt.tight_layout()
plt.show()
