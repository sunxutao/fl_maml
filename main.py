import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import logging
import os
import sys
from fl import FL
from utils import load_data
from utils import create_exp_dir
#from maml import MAML

# hyper-parameters
parser = argparse.ArgumentParser("FL_MAML")
parser.add_argument('--fraction', type=float, default=0.005, help='fraction of selected clients') # 0.01 0.005
parser.add_argument('--num_rounds', type=int, default=100, help='number of communication rounds')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs') # 1 10 20
parser.add_argument('--batch_size', type=int, default=10, help='batch size') # 10 50
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--model', type=str, default='MLP', help='model') # MLP or LeNet
parser.add_argument('--gpu', type=str, default='cuda', help='use gpu') # cuda or cpu
parser.add_argument('--log_path', type=str, default='log', help='log folder name')
args = parser.parse_args()

args.device = torch.device(args.gpu) # environment variable: CUDA_VISIABLE_DEVICES
args.save_path = os.path.join(os.getcwd(), args.log_path)

# set up logger
create_exp_dir(args.save_path)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S')
fh = logging.FileHandler(os.path.join(args.save_path, '{}_E{}_B{}_C{}.txt'
                    .format(args.model, args.num_epochs, args.batch_size, args.fraction))) # Model_E_B_C.txt
fh.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger()
logger.addHandler(fh)

# load data (client: 3383 / train: 341873 / test: 40832)
d_train = load_data('fed_emnist_digitsonly_train.h5', args)
d_test = load_data('fed_emnist_digitsonly_test.h5', args)

if __name__ == '__main__':
    fl_train_acc, fl_test_acc, fl_train_loss, fl_test_loss = FL(d_train, d_test, args)