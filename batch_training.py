import os
import argparse
from pathlib import Path
import shutil

class color:
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
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")

args = parser.parse_args()



session_name = args.name
session_path = f'./output/batch_training/{session_name}/'
if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            print(f'Error in session creation ({session_path}).')
            exit()
    else:
        print(f'Session already exists ({session_path}), overwrite the session? (y/n): ', end='')
        force_write = input()
        if force_write == 'y':
            print('')
            try:
                shutil.rmtree(session_path)
                Path(session_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f'Error in session creation ({session_path}).')
                exit()
        else:
            exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f'Error in session creation ({session_path}).')
        exit()

print(f'{color.BOLD}Starting {color.UNDERLINE}batch training{color.END}{color.BOLD} session \'{session_name}\'{color.END}')



sessions_list = open(f'./output/batch_training/{session_name}/sessions_list.txt', "w")
# log_path = f'./output/batch_training/{session_name}/tail_log.txt'



batch_dic = {
    1:{
    'name': 'NLRAV_20_20_100',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 100
    },
    2:{
    'name': 'NLRAV_20_20_25',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 25
    },
    3:{
    'name': 'NLRAV_20_20_10',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 10
    },
    4:{
    'name': 'NLRAV_20_4_100',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 100
    },
    5:{
    'name': 'NLRAV_20_4_25',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 25
    },
    6:{
    'name': 'NLRAV_20_4_10',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 10
    },
    7:{
    'name': 'NLRAV_4_4_100',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 100
    },
    8:{
    'name': 'NLRAV_4_4_25',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 25
    },
    9:{
    'name': 'NLRAV_4_4_10',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 10
    },
    10:{
    'name': 'NLRAV_4_4_5',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 5
    },
    11:{
    'name': 'NLRAV_2_2_5',
    'epoch' : 200,
    'dataset' : 'NLRAV',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 2,
    'conv2of' : 2,
    'foutdim' : 5
    },
    12:{
    'name': 'NSVFQ_20_20_100',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 100
    },
    13:{
    'name': 'NSVFQ_20_20_25',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 25
    },
    14:{
    'name': 'NSVFQ_20_20_10',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 20,
    'foutdim' : 10
    },
    15:{
    'name': 'NSVFQ_20_4_100',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 100
    },
    16:{
    'name': 'NSVFQ_20_4_25',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 25
    },
    17:{
    'name': 'NSVFQ_20_4_10',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 20,
    'conv2of' : 4,
    'foutdim' : 10
    },
    18:{
    'name': 'NSVFQ_4_4_100',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 100
    },
    19:{
    'name': 'NSVFQ_4_4_25',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 25
    },
    20:{
    'name': 'NSVFQ_4_4_10',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 10
    },
    21:{
    'name': 'NSVFQ_4_4_5',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 4,
    'conv2of' : 4,
    'foutdim' : 5
    },
    22:{
    'name': 'NSVFQ_2_2_5',
    'epoch' : 200,
    'dataset' : 'NSVFQ',
    'split' : 0.7,
    'augmentation' : [0,1],
    'randomseed' : 0,
    'batchsize' : 32,
    'norm' : False,
    'indim' : 198,
    'ksize' : 7,
    'conv1of' : 2,
    'conv2of' : 2,
    'foutdim' : 5
    }
}



for d in batch_dic:
    print(f'{color.BOLD}{color.UNDERLINE}{color.CYAN}\n\n\n--- SESSIONS LEFT: {len(batch_dic) - list(batch_dic).index(d)} ---\n\n\n{color.END}')
    sessions_list.write(f"{batch_dic[d]['name']}\n")
    os.system(f"python3 training.py -n {batch_dic[d]['name']} -e {batch_dic[d]['epoch']} -d {batch_dic[d]['dataset']} -s {batch_dic[d]['split']} -a {batch_dic[d]['augmentation'][0]} {batch_dic[d]['augmentation'][1]} -r {batch_dic[d]['randomseed']} -b {batch_dic[d]['batchsize']} --indim {batch_dic[d]['indim']} --ksize {batch_dic[d]['ksize']} --conv1of {batch_dic[d]['conv1of']} --conv2of {batch_dic[d]['conv2of']} --foutdim {batch_dic[d]['foutdim']} {'--norm ' if batch_dic[d]['norm'] else ''}")
print(f'{color.BOLD}{color.UNDERLINE}{color.CYAN}\n\n\n--- NO SESSIONS LEFT ---\n\n\n{color.END}')

sessions_list.close()

print(f'{color.BOLD}Ending {color.UNDERLINE}batch training{color.END}{color.BOLD} session \'{session_name}\'{color.END}')
