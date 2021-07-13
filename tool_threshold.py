import os
import argparse
from pathlib import Path
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', dest='name', required=True, help="session name")
parser.add_argument('-m','--multiplier', dest='multiplier', required=True, help="threshold multiplier")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")

args = parser.parse_args()



session_name = args.name
session_path = f'./output/threshold/{session_name}/'
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



data_names = ['100', '101', '102', '103', '104', '105', '106', '107',
              '108', '109', '111', '112', '113', '114', '115', '116',
              '117', '118', '119', '121', '122', '123', '124', '200',
              '201', '202', '203', '205', '207', '208', '209', '210',
              '212', '213', '214', '215', '217', '219', '220', '221',
              '222', '223', '228', '230', '231', '232', '233', '234']

data_threshold = {
    '100' : 400000,
    '101' : 250000,
    '102' : 300000,
    '103' : 300000,
    '104' : 600000,
    '105' : 700000,
    '106' : 300000,
    '107' : 1500000,
    '108' : 200000,
    '109' : 200000,
    '111' : 200000,
    '112' : 200000,
    '113' : 4000000,
    '114' : 200000,
    '115' : 200000,
    '116' : 1250000,
    '117' : 200000,
    '118' : 850000,
    '119' : 1000000,
    '121' : 200000,
    '122' : 200000,
    '123' : 220000,
    '124' : 1000000,
    '200' : 850000,
    '201' : 145000,
    '202' : 200000,
    '203' : 550000,
    '205' : 200000,
    '207' : 240000,
    '208' : 220000,
    '209' : 540000,
    '210' : 192500,
    '212' : 1100000,
    '213' : 1100000,
    '214' : 550000,
    '215' : 440000,
    '217' : 825000,
    '219' : 825000,
    '220' : 1100000,
    '221' : 185000,
    '222' : 165000,
    '223' : 192500,
    '228' : 260000,
    '230' : 825000,
    '231' : 825000,
    '232' : 209000,
    '233' : 715000,
    '234' : 192500
}

th_multiplier = float(args.multiplier)
for d in data_names:
    file = open(f'{session_path}{d}_th.txt', 'w')
    file.write(f'{int(data_threshold[d]*th_multiplier)}')
    file.close()
