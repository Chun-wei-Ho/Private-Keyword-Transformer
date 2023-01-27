import argparse
import os
import numpy as np

import multiprocessing as mp

def get_acc(logfile):
    with open(logfile, 'r') as f:
        line = f.readlines()[-3]
    if 'Final test accuracy' not in line: return
    result = line.split(' ')[-2].strip('%')
    return float(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    logs = [os.path.join(args.input_dir, d, 'train.log')
        for d in os.listdir(args.input_dir)]
    logs = [i for i in logs if os.path.isfile(i)]

    with mp.Pool() as pool:
        acc = pool.map(get_acc, logs)
        acc = [i for i in acc if i is not None]

    print(args.input_dir)
    print(f"Number of finished processes: {len(acc)}")
    print(f"Mean ACC: {np.mean(acc)}")
    print(f"Std ACC: {np.std(acc)}")
