"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import subprocess
import time
from tqdm import tqdm
import multiprocessing
import sys

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    
def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()
    
def get_best_gpu(force = None):
    if force is not None:return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace('MiB', '').replace('memory.free', '').split('\n')
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    if len(a) == 0: 
        print("no GPU available. Using CPU...")
        return torch.device('cpu')
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',ss[best + 1])
    return torch.device('cuda:'+str(best))

def check_modification():
    runtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime) 
    return runtime
