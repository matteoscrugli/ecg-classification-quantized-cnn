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

# parser = argparse.ArgumentParser()
# parser.add_argument('-n','--name', dest='name', required=True, help="session name")
# parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")
#
# args = parser.parse_args()



# session_name = args.name
# session_path = "output/peak_detector/"+session_name+"/"
# if os.path.isdir(session_path):
#     if args.overwrite:
#         try:
#             shutil.rmtree(session_path)
#             Path(session_path).mkdir(parents=True, exist_ok=True)
#         except OSError:
#             print("Error in session creation ("+session_path+").")
#             exit()
#     else:
#         print("Session already exists ("+session_path+"), overwrite the session? (y/n): ", end='')
#         force_write = input()
#         if force_write == "y":
#             print('')
#             try:
#                 shutil.rmtree(session_path)
#                 Path(session_path).mkdir(parents=True, exist_ok=True)
#             except OSError:
#                 print("Error in session creation ("+session_path+").")
#                 exit()
#         else:
#             exit()
# else:
#     try:
#         Path(session_path).mkdir(parents=True, exist_ok=True)
#     except OSError:
#         print("Error in session creation ("+session_path+").")
#         exit()



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




data_names = ['100', '101', '102', '103', '104', '105', '106', '107',
              '108', '109', '111', '112', '113', '114', '115', '116',
              '117', '118', '119', '121', '122', '123', '124', '200',
              '201', '202', '203', '205', '207', '208', '209', '210',
              '212', '213', '214', '215', '217', '219', '220', '221',
              '222', '223', '228', '230', '231', '232', '233', '234']



float2int_mul = 10000;
wid = 99


Y = []
F = []
P = []

Y_NLRAV = []
F_NLRAV = []
P_NLRAV = []

Y_NSVFQ = []
F_NSVFQ = []
P_NSVFQ = []



labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
NLRAV_labels = ['N', 'L', 'R', 'A', 'V']
NSVFQ_labels = ['N', 'S', 'V', 'F', 'Q']
NSVFQ_sub_labels = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
       'A':'S', 'a':'S', 'J':'S', 'S':'S',
       'V':'V', 'E':'V',
       'F':'F',
       '/':'Q', 'f':'Q', 'Q':'Q'}

allstat_min = 0
allstat_max = 0



session_path = './output/dataset/raw_text/'
if os.path.isdir(session_path):
    try:
        shutil.rmtree(session_path)
        Path(session_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

# session_path_text = './output/dataset/raw_peaks/'
# if os.path.isdir(session_path):
#     try:
#         shutil.rmtree(session_path)
#         Path(session_path).mkdir(parents=True, exist_ok=True)
#     except OSError:
#         print("Error in session creation ("+session_path+").")
#         exit()
# else:
#     try:
#         Path(session_path).mkdir(parents=True, exist_ok=True)
#     except OSError:
#         print("Error in session creation ("+session_path+").")
#         exit()

time.sleep(6)



f_allstat = open('./output/dataset/raw_text/all_stat.txt', "w")

printProgressBar(0, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)
for d in data_names:
    f_data = open('./output/dataset/raw_text/'+d+'_data.txt', "w")
    f_intdata = open('./output/dataset/raw_text/'+d+'_intdata.txt', "w")
    f_labels = open('./output/dataset/raw_text/'+d+'_label.txt', "w")
    f_labelspos = open('./output/dataset/raw_text/'+d+'_labelpos.txt', "w")
    f_labels_NLRAV = open('./output/dataset/raw_text/'+d+'_label_NLRAV.txt', "w")
    f_labelspos_NLRAV = open('./output/dataset/raw_text/'+d+'_labelpos_NLRAV.txt', "w")
    f_labels_NSVFQ = open('./output/dataset/raw_text/'+d+'_label_NSVFQ.txt', "w")
    f_labelspos_NSVFQ = open('./output/dataset/raw_text/'+d+'_labelpos_NSVFQ.txt', "w")
    f_stat = open('./output/dataset/raw_text/'+d+'_stat.txt', "w")
    r = wfdb.rdrecord('./dataset/raw/'+d)
    ann = wfdb.rdann('./dataset/raw/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    sig = np.array(r.p_signal[:,0])
    intsig = np.array(r.p_signal[:,0])
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample
    beat_len = len(sym)
    stat_min = sig[0]
    stat_max = sig[0]
    for i in range(sig_len):
        f_data.write(str(sig[i])+'\n')
        intsig[i]=int(sig[i]*float2int_mul)
        f_intdata.write(str(intsig[i])+'\n')
        if sig[i] < stat_min:
            stat_min = sig[i]
        if sig[i] > stat_max:
            stat_max = sig[i]
    for i in range(beat_len):
        if pos[i]-wid>=0 and pos[i]+wid<=sig_len and sym[i] in labels:
        # if pos[i]-wid>=0 and pos[i]+wid<=sig_len:
            f_labels.write(str(sym[i])+'\n')
            f_labelspos.write(str(pos[i])+'\n')

            # a = sig[pos[i]-wid:pos[i]+wid]
            # if len(a) != 2*wid:
            #     print("Length error")
            #     continue
            # Y.append(labels.index(sym[i]))
            # F.append(data_names.index(d))
            # P.append(pos[i])

            if sym[i] in NLRAV_labels:
                f_labels_NLRAV.write(str(sym[i])+'\n')
                f_labelspos_NLRAV.write(str(pos[i])+'\n')

                # Y_NLRAV.append(NLRAV_labels.index(sym[i]))
                # F_NLRAV.append(data_names.index(d))
                # P_NLRAV.append(pos[i])

            f_labels_NSVFQ.write(str(NSVFQ_sub_labels[sym[i]])+'\n')
            f_labelspos_NSVFQ.write(str(pos[i])+'\n')

            # Y_NSVFQ.append(NSVFQ_labels.index(NSVFQ_sub_labels[sym[i]]))
            # F_NSVFQ.append(data_names.index(d))
            # P_NSVFQ.append(pos[i])
    if d == data_names[0]:
        allstat_min = stat_min
        allstat_max = stat_max
    else:
        if stat_min < allstat_min:
            allstat_min = stat_min
        if stat_max > allstat_max:
            allstat_max = stat_max

    f_stat.write(str(stat_min)+'\n'+str(stat_max))
    f_data.close()
    f_intdata.close()
    f_labels.close()
    f_labelspos.close()
    f_labels_NLRAV.close()
    f_labelspos_NLRAV.close()
    f_labels_NSVFQ.close()
    f_labelspos_NSVFQ.close()
    f_stat.close()
    printProgressBar(data_names.index(d) + 1, len(data_names), prefix = 'Files analyzed:', suffix = '', length = 60)

f_allstat.write(str(allstat_min)+'\n'+str(allstat_max))
f_allstat.close()



# t_done = False
# t_dict = {'prefix' : 'Exporting dataset peaks: '}
# t = threading.Thread(target=animate, kwargs=t_dict)
# t.start()
#
# Y = np.array(Y)
# F = np.array(F)
# P = np.array(P)
#
# Y_NLRAV = np.array(Y_NLRAV)
# F_NLRAV = np.array(F_NLRAV)
# P_NLRAV = np.array(P_NLRAV)
#
# Y_NSVFQ = np.array(Y_NSVFQ)
# F_NSVFQ = np.array(F_NSVFQ)
# P_NSVFQ = np.array(P_NSVFQ)
#
#
#
# temp_path = './output/dataset/raw_peaks/ordered/'
# if os.path.isdir(temp_path):
#     shutil.rmtree(temp_path)
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
# else:
#     Path(temp_path).mkdir(parents=True, exist_ok=True)
#
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
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NLRAV, output_file)
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F_NLRAV, output_file)
# with open("./output/dataset/raw_peaks/ordered_NLRAV/dataset_position_in_file.pickle", "wb") as output_file:
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
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks/ordered_NSVFQ/dataset_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NSVFQ, output_file)
#
#
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
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_data.pickle", "wb") as output_file:
#     pk.dump(X_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F_NLRAV, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NLRAV/dataset_position_in_file.pickle", "wb") as output_file:
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
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_data.pickle", "wb") as output_file:
#     pk.dump(X_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_labels.pickle", "wb") as output_file:
#     pk.dump(Y_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_file.pickle", "wb") as output_file:
#     pk.dump(F_NSVFQ, output_file)
# with open("./output/dataset/raw_peaks_aug/ordered_NSVFQ/dataset_position_in_file.pickle", "wb") as output_file:
#     pk.dump(P_NSVFQ, output_file)
#
# t_done = True
# time.sleep(0.1)
#
# print('end')
# exit()
