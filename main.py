import torch
import torch.nn as nn
import argparse
import logging
import os
import sys
from fl import FL
from fl_lstm import FL_LSTM
from utils import load_data, create_exp_dir, Dictionary, lstm_data_process, load_poisoned_data
import time
import gc

if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser("FL_MAML")
    parser.add_argument('--model', type=str, default='LeNet',
                        help='model')  # MLP or LeNet or LSTM
    parser.add_argument('--gpu', type=str, default='cuda',
                        help='use gpu or cpu')  # cuda or cpu
    parser.add_argument('--log_path', type=str, default='log',
                        help='log folder name')

    # fraction related parameters
    parser.add_argument('--fraction', type=float, default=0.005,
                        help='fraction of selected clients per round')  # 0.01 0.005
    parser.add_argument('--lstm_fraction', type=float, default=0.05,
                        help='fraction of selected clients per round')  # 0.1 0.05 0.02
    parser.add_argument('--poisoning_fraction', type=float, default=0.2,
                        help='fraction of poisoned clients')  # 0.1 0.2 0.3 0.4

    # robust method parameters
    parser.add_argument('--robust_method', type=str, default='none',
                        help='robust method type')  # none, krum, incentive

    # epoch related parameters
    parser.add_argument('--num_rounds', type=int, default=500,
                        help='number of communication rounds')
    parser.add_argument('--train_epochs', type=int, default=5,
                        help='number of training epochs of FL')  # 1 10 20

    # batch size
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')  # 10 50

    # learning rate related parameters
    parser.add_argument('--train_lr', type=float, default=0.05,
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
    args.device = torch.device(args.gpu)  # environment variable: CUDA_VISIBLE_DEVICES
    args.save_path = os.path.join(os.getcwd(), args.log_path)
    args.loss_func = nn.CrossEntropyLoss().to(args.device)
    args.local_interval = 10

    # set up logger
    create_exp_dir(args.save_path)
    # log_format = '%(asctime)s %(message)s'
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S')
    time_stamp = time.time()
    time_arr = time.localtime(time_stamp)
    # Time_Algorithm_Model_Epoch_Batch_Fraction.txt
    # if args.model == 'LSTM':
    #     fh = logging.FileHandler(os.path.join(args.save_path, '{}{}{}{}{}{}_{}_E{}_B{}_C{}.txt'
    #                                           .format(time_arr[0], str(time_arr[1]).zfill(2), str(time_arr[2]).zfill(2),
    #                                                   str(time_arr[3]).zfill(2), str(time_arr[4]).zfill(2),
    #                                                   str(time_arr[5]).zfill(2),
    #                                                   args.model, args.train_epochs, args.batch_size,
    #                                                   args.lstm_fraction)))
    # else:
    #     fh = logging.FileHandler(os.path.join(args.save_path, '{}{}{}{}{}{}_{}_E{}_B{}_C{}.txt'
    #                                           .format(time_arr[0], str(time_arr[1]).zfill(2), str(time_arr[2]).zfill(2),
    #                                                   str(time_arr[3]).zfill(2), str(time_arr[4]).zfill(2),
    #                                                   str(time_arr[5]).zfill(2),
    #                                                   args.model, args.train_epochs, args.batch_size, args.fraction)))
    # fh.setFormatter(logging.Formatter(log_format))
    # logger = logging.getLogger()
    # logger.addHandler(fh)

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
        d_train, d_test = lstm_data_process(d_train, d_test, corpus, args)
        args.ntokens = len(corpus)
        args.corpus = corpus

        FL_LSTM(d_train, d_test, args)

    else:
        # load MNIST data (client: 3383 / train: 341873 / test: 40832)
        # d_train = load_poisoned_data('data/fed_emnist_digitsonly_train.h5', args)
        # d_train = load_data('data/fed_emnist_digitsonly_train.h5', args)
        # d_test = load_data('data/fed_emnist_digitsonly_test.h5', args)

        num_avg = 50
        file_open = open("poison_fraction_results.txt", "w")
        # file_open.write("*****Begin normal situation*****\n")
        # file_open.flush()
        #
        # # normal situation
        # sum_normal = 0.0
        # for i in range(num_avg):
        #     d_train = load_data('data/fed_emnist_digitsonly_train.h5', args)
        #     d_test = load_data('data/fed_emnist_digitsonly_test.h5', args)
        #     current_test_accuracy = FL(d_train, d_test, args)
        #     file_open.write("iteration " + str(i) + " test accuracy = " + str(current_test_accuracy) + "\n")
        #     file_open.flush()
        #     sum_normal += current_test_accuracy
        #     del d_train, d_test
        #     gc.collect()
        #
        # avg_normal = sum_normal / num_avg
        # file_open.write("normal situation: batch_size = " + str(args.batch_size) + ", client_fraction = " + str(args.fraction) + ", train_epochs = " + str(args.train_epochs) +
        #                 ", iteration_num = " + str(args.num_rounds) + ", test accuracy = " + str(avg_normal) + "\n")
        # file_open.flush()
        #
        # file_open.write("*****Begin poisoning situation, with none robust method*****\n")
        # file_open.flush()

        # poisoning situation with none robust method
        # poisoning_fraction = [0.1, 0.2, 0.3]
        poisoning_fraction = [0.3]

        for fraction in poisoning_fraction:
            file_open.write("poisoning fraction = " + str(fraction) + "\n")
            file_open.flush()
            args.poisoning_fraction = fraction
            sum_poisoning = 0.0
            for i in range(num_avg):
                d_train_poisoning = load_poisoned_data('data/fed_emnist_digitsonly_train.h5', args)
                d_test_poisoning = load_data('data/fed_emnist_digitsonly_test.h5', args)
                current_poison_test_accuracy = FL(d_train_poisoning, d_test_poisoning, args)
                file_open.write("iteration " + str(i) + " test accuracy = " + str(current_poison_test_accuracy) + "\n")
                file_open.flush()
                sum_poisoning += current_poison_test_accuracy
                del d_train_poisoning, d_test_poisoning
                gc.collect()
            avg_poisoning = sum_poisoning / num_avg
            file_open.write(
                "poisoned situation: batch_size = " + str(args.batch_size) + ", client_fraction = " + str(args.fraction) + ", train_epochs = " + str(args.train_epochs) +
                ", iteration_num = " + str(args.num_rounds) + ", poisoning_fraction = " + str(args.poisoning_fraction) + ", test accuracy = " + str(avg_poisoning) + "\n")
            file_open.flush()

        file_open.write("*****Begin poisoning situation, with krum robust method*****\n")
        file_open.flush()
        args.robust_method = 'krum'

        # poisoning situation with krum robust method
        for fraction in poisoning_fraction:
            file_open.write("poisoning fraction = " + str(fraction) + "\n")
            file_open.flush()
            args.poisoning_fraction = fraction
            sum_poisoning = 0.0
            for i in range(num_avg):
                d_train_poisoning = load_poisoned_data('data/fed_emnist_digitsonly_train.h5', args)
                d_test_poisoning = load_data('data/fed_emnist_digitsonly_test.h5', args)
                current_poison_test_accuracy = FL(d_train_poisoning, d_test_poisoning, args)
                file_open.write("iteration " + str(i) + " test accuracy = " + str(current_poison_test_accuracy) + "\n")
                file_open.flush()
                sum_poisoning += current_poison_test_accuracy
                del d_train_poisoning, d_test_poisoning
                gc.collect()
            avg_poisoning = sum_poisoning / num_avg
            file_open.write(
                "poisoned situation: batch_size = " + str(args.batch_size) + ", client_fraction = " + str(args.fraction) + ", train_epochs = " + str(args.train_epochs) +
                ", iteration_num = " + str(args.num_rounds) + ", poisoning_fraction = " + str(args.poisoning_fraction) + ", test accuracy = " + str(avg_poisoning) + "\n")
            file_open.flush()

        file_open.write("*****Begin poisoning situation, with incentive robust method*****\n")
        file_open.flush()
        args.robust_method = 'incentive'

        # poisoning situation with incentive robust method
        for fraction in poisoning_fraction:
            file_open.write("poisoning fraction = " + str(fraction) + "\n")
            file_open.flush()
            args.poisoning_fraction = fraction
            sum_poisoning = 0.0
            for i in range(num_avg):
                d_train_poisoning = load_poisoned_data('data/fed_emnist_digitsonly_train.h5', args)
                d_test_poisoning = load_data('data/fed_emnist_digitsonly_test.h5', args)
                current_poison_test_accuracy = FL(d_train_poisoning, d_test_poisoning, args)
                file_open.write("iteration " + str(i) + " test accuracy = " + str(current_poison_test_accuracy) + "\n")
                file_open.flush()
                sum_poisoning += current_poison_test_accuracy
                del d_train_poisoning, d_test_poisoning
                gc.collect()
            avg_poisoning = sum_poisoning / num_avg
            file_open.write(
                "poisoned situation: batch_size = " + str(args.batch_size) + ", client_fraction = " + str(
                    args.fraction) + ", train_epochs = " + str(args.train_epochs) +
                ", iteration_num = " + str(args.num_rounds) + ", poisoning_fraction = " + str(
                    args.poisoning_fraction) + ", test accuracy = " + str(avg_poisoning) + "\n")
            file_open.flush()

        file_open.write("*****Begin normal situation*****\n")
        file_open.flush()

        # normal situation
        sum_normal = 0.0
        for i in range(num_avg):
            d_train = load_data('data/fed_emnist_digitsonly_train.h5', args)
            d_test = load_data('data/fed_emnist_digitsonly_test.h5', args)
            current_test_accuracy = FL(d_train, d_test, args)
            file_open.write("iteration " + str(i) + " test accuracy = " + str(current_test_accuracy) + "\n")
            file_open.flush()
            sum_normal += current_test_accuracy
            del d_train, d_test
            gc.collect()

        avg_normal = sum_normal / num_avg
        file_open.write("normal situation: batch_size = " + str(args.batch_size) + ", client_fraction = " + str(
            args.fraction) + ", train_epochs = " + str(args.train_epochs) +
                        ", iteration_num = " + str(args.num_rounds) + ", test accuracy = " + str(avg_normal) + "\n")
        file_open.flush()

        file_open.write("*****Begin poisoning situation, with none robust method*****\n")
        file_open.flush()

        file_open.close()