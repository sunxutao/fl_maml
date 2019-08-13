import torch
import torch.nn as nn
import argparse
import logging
import os
import sys
from fl import FL
from fl_lstm import FL_LSTM
from utils import load_data, create_exp_dir, Dictionary, lstm_data_process
import time


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser("FL_MAML")
    parser.add_argument('--model', type=str, default='LeNet',
                        help='model') # MLP or LeNet or LSTM
    parser.add_argument('--gpu', type=str, default='cuda',
                        help='use gpu or cpu') # cuda or cpu
    parser.add_argument('--log_path', type=str, default='log',
                        help='log folder name')

    # fraction related parameters
    parser.add_argument('--fraction', type=float, default=0.005,
                        help='fraction of selected clients per round') # 0.01 0.005
    parser.add_argument('--lstm_fraction', type=float, default=0.05,
                        help='fraction of selected clients per round')  # 0.1 0.05 0.02

    # epoch related parameters
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of communication rounds')
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='number of training epochs of FL') # 1 10 20

    # batch size
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size') # 10 50

    # learning rate related parameters
    parser.add_argument('--train_lr', type=float, default=0.1,
                        help='train learning rate')
    parser.add_argument('--lstm_lr', type=float, default=20,
                        help='learning rate for LSTM')

    # LSTM parameters
    parser.add_argument('--bptt', type=int, default=10,
                        help='sequence length')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')

    args = parser.parse_args()

    if args.gpu == 'cuda' and torch.cuda.is_available() == False:
        print('No GPU available, training on CPU')
        args.gpu = 'cpu'
    args.device = torch.device(args.gpu) # environment variable: CUDA_VISIBLE_DEVICES
    args.save_path = os.path.join(os.getcwd(), args.log_path)
    args.loss_func = nn.CrossEntropyLoss().to(args.device)
    args.local_interval = 10

    # set up logger
    create_exp_dir(args.save_path)
    #log_format = '%(asctime)s %(message)s'
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S')
    time_stamp = time.time()
    time_arr = time.localtime(time_stamp)
    # Time_Algorithm_Model_Epoch_Batch_Fraction.txt
    if args.model == 'LSTM':
        fh = logging.FileHandler(os.path.join(args.save_path, '{}{}{}{}{}{}_{}_E{}_B{}_C{}.txt'
                                              .format(time_arr[0], str(time_arr[1]).zfill(2), str(time_arr[2]).zfill(2),
                                                      str(time_arr[3]).zfill(2), str(time_arr[4]).zfill(2),
                                                      str(time_arr[5]).zfill(2),
                                                      args.model, args.train_epochs, args.batch_size,
                                                      args.lstm_fraction)))
    else:
        fh = logging.FileHandler(os.path.join(args.save_path, '{}{}{}{}{}{}_{}_E{}_B{}_C{}.txt'
                                              .format(time_arr[0],str(time_arr[1]).zfill(2),str(time_arr[2]).zfill(2),
                                                      str(time_arr[3]).zfill(2),str(time_arr[4]).zfill(2),str(time_arr[5]).zfill(2),
                                                      args.model,args.train_epochs,args.batch_size,args.fraction)))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info('*' * 20 + ' Parameters ' + '*' * 20)

    if args.model == 'LSTM':
        logging.info('train lr: {}'.format(args.lstm_lr))
    else:
        logging.info('train lr: {}'.format(args.train_lr))
    logging.info('batch size: {}'.format(args.batch_size))
    logging.info('epochs: {}'.format(args.train_epochs))

    logging.info('*' * 52)

    if args.model == 'LSTM':
        # load shakespeare data (client: 715)
        d_train = load_data('data/shakespeare_train.h5', args)
        d_test = load_data('data/shakespeare_test.h5', args)

        # preprocess data and construct a dictionary for whole data
        corpus = Dictionary()
        d_train, d_test = lstm_data_process(d_train, d_test, corpus,args)
        args.ntokens = len(corpus)
        args.corpus = corpus

        FL_LSTM(d_train, d_test, args)

    else:
        # load MNIST data (client: 3383 / train: 341873 / test: 40832)
        d_train = load_data('data/fed_emnist_digitsonly_train.h5', args)
        d_test = load_data('data/fed_emnist_digitsonly_test.h5', args)

        FL(d_train, d_test, args)
