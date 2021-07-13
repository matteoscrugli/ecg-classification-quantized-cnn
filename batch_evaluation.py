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
parser.add_argument('-p','--peak', dest='peak', help="peak detection session path")
parser.add_argument('-t','--train', dest='train', help=".txt list of training sessions")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")

args = parser.parse_args()



session_name = args.name
session_path = f'./output/batch_evaluation/{session_name}/'
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

print(f'{color.BOLD}Starting {color.UNDERLINE}batch evaluation{color.END}{color.BOLD} session \'{session_name}\'{color.END}')



if args.peak == None:
    session_peak_path = 'output/peakdetector/deafault/'
else:
    session_peak_path = args.peak

if args.train == None:
    batch_dic = {
        0:{
            'name':'default',
            'peak':'output/peakdetector/default',
            'train':f'output/train/default'
        }
    }
elif args.train == 'all':
    session_train_list = [dI for dI in os.listdir('output/train/') if os.path.isdir(os.path.join('output/train/',dI))]
    batch_dic = {}
    for line in session_train_list:
        batch_dic[len(batch_dic)] = {
            'name':f'{line}',
            'peak':f'{session_peak_path}',
            'train':f'output/train/{line}'
            }
else:
    f_list = open(args.train, 'r')
    session_train_list = f_list.readlines()
    batch_dic = {}
    for line in session_train_list:
        batch_dic[len(batch_dic)] = {
            'name':f'{line}'.replace("\n",""),
            'peak':f'{session_peak_path}',
            'train':f'output/train/{line}'.replace("\n","")
            }

sessions_list = open(f'./output/batch_evaluation/{session_name}/sessions_list.txt', "w")






for d in batch_dic:
    print(f'{color.BOLD}{color.UNDERLINE}{color.CYAN}\n\n\n--- SESSIONS LEFT: {len(batch_dic) - list(batch_dic).index(d)} ---\n\n\n{color.END}')
    sessions_list.write(f"{batch_dic[d]['name']}\n")
    os.system(f"python3 evaluation.py -n {batch_dic[d]['name']} -p {batch_dic[d]['peak']} -t {batch_dic[d]['train']}")
print(f'{color.BOLD}{color.UNDERLINE}{color.CYAN}\n\n\n--- NO SESSIONS LEFT ---\n\n\n{color.END}')

sessions_list.close()

print(f'{color.BOLD}Ending {color.UNDERLINE}batch evaluation{color.END}{color.BOLD} session \'{session_name}\'{color.END}')
