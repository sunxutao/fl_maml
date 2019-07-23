import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import logging
import os
import sys
from fl import FL
from utils import load_data, split_data, create_exp_dir
from fedreptile import FR


# parameters
parser = argparse.ArgumentParser("FL_MAML")
parser.add_argument('--algo', type=str, default='FL', help='Algorithm: Federated Learning or FedReptile') # FL or FR
parser.add_argument('--model', type=str, default='LeNet', help='model') # MLP or LeNet
parser.add_argument('--gpu', type=str, default='cuda', help='use gpu or cpu') # cuda or cpu
parser.add_argument('--log_path', type=str, default='log', help='log folder name')

# client related parameters
parser.add_argument('--fraction', type=float, default=0.005, help='fraction of selected clients per round') # 0.01 0.005
parser.add_argument('--fraction_t', type=float, default=0.9, help='fraction of support clients for meta learning')

# epoch related parameters
parser.add_argument('--num_rounds', type=int, default=1500, help='number of communication rounds')
parser.add_argument('--train_epochs', type=int, default=10, help='number of training epochs') # 1 10 20
parser.add_argument('--local_epochs', type=int, default=10, help='number of localization epochs')

# batch size
parser.add_argument('--batch_size', type=int, default=10, help='batch size') # 10 50

# learning rate related parameters
parser.add_argument('--train_lr', type=float, default=0.1, help='train learning rate')
parser.add_argument('--local_lr', type=float, default=0.02, help='localization learning rate')
parser.add_argument('--global_lr', type=float, default=1, help='outer learning rate for globalization')

args = parser.parse_args()

args.device = torch.device(args.gpu) # environment variable: CUDA_VISIBLE_DEVICES
args.save_path = os.path.join(os.getcwd(), args.log_path)
args.loss_func = nn.CrossEntropyLoss()
args.test_client_interval = 10

# set up logger
create_exp_dir(args.save_path)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S')
fh = logging.FileHandler(os.path.join(args.save_path, '{}_{}_E{}_B{}_C{}.txt'
                .format(args.algo, args.model, args.train_epochs, args.batch_size, args.fraction))) # Model_E_B_C.txt
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger()
logger.addHandler(fh)

# load data (client: 3383 / train: 341873 / test: 40832)
d_train = load_data('fed_emnist_digitsonly_train.h5', args)
d_test = load_data('fed_emnist_digitsonly_test.h5', args)

# split data into two parts: support client / test client
support_train, support_test, test_train, test_test = split_data(d_train, d_test, args)

if __name__ == '__main__':
    if args.algo == 'FL':
        FL(support_train, support_test, test_train, test_test, args)
    elif args.algo == 'FR':
        FR(support_train, support_test, test_train, test_test, args)