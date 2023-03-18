import os, sys
import numpy as np

import time
import torch
import scipy.io
import matplotlib.pyplot as plt
from helpers import *
from MLP import *

import time

from load_data import *
from run_sdf import Runner
import logging, argparse
import scipy.io

   
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.is_continue, write_config=False)
    
    runner.set_params()
    runner.validate_mesh(threshold=0.1)
