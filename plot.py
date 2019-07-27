import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os

if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser("Plot")
    parser.add_argument('--fl_log_path', type=str, default=None, help='FL log path for plotting')
    parser.add_argument('--fr_log_path', type=str, default=None, help='FR log path for plotting')
    parser.add_argument('--num_rounds', type=int, default=100, help='range of rounds for observation')
    args = parser.parse_args()
    args.local_interval = 10

    args.fl_log_path = '/home/sunxutao/logs/20190727112302_FL_LeNet_E1_B10_C0.005.txt'
    args.fr_log_path = '/home/sunxutao/logs/20190727112409_FR_LeNet_E1_B10_C0.005.txt'
    # args.log_path = 'log/20190724210418_FR_LeNet_E1_B10_C0.005.txt'


    # args.fl_log_path = os.path.join(os.getcwd(), args.log_path)
    pattern1 = re.compile('^round\s+\d+: support_train_acc (\d+\.?\d*), support_train_loss (\d+\.?\d*), '
                          'support_test_acc (\d+\.?\d*), support_test_loss (\d+\.?\d*)')
    pattern2 = re.compile('^initial_acc (\d+\.?\d*), initial_loss (\d+\.?\d*)')
    pattern3 = re.compile('^localization_acc (\d+\.?\d*), localization_loss (\d+\.?\d*)')
    pattern4 = re.compile('^epoch:\s+\d+: test acc: (\d+\.?\d*), test loss: (\d+\.?\d*)')

    if args.fl_log_path:
        m = re.match('.*B(\d+).*', args.fl_log_path)
        fl_batch_size = m.group(1)
        m = re.match('.*E(\d+).*', args.fl_log_path)
        fl_num_epochs = m.group(1)
        m = re.match('.*C(0.\d+).*', args.fl_log_path)
        fl_fraction = m.group(1)

        fl_support_train_acc = []
        fl_support_train_loss = []
        fl_support_test_acc = []
        fl_support_test_loss = []
        fl_test_initial_acc = []
        fl_test_initial_loss = []
        fl_test_train_acc = []
        fl_test_train_loss = []
        fl_test_test_acc = []
        fl_test_test_loss = []

        reader = open(args.fl_log_path, 'r')
        while True:
            line = reader.readline()
            if len(line) == 0:
                break
            line = line.rstrip()
            if re.match(pattern1, line):
                m = re.match(pattern1, line)
                fl_support_train_acc.append(float(m.group(1)))
                fl_support_train_loss.append(float(m.group(2)))
                fl_support_test_acc.append(float(m.group(3)))
                fl_support_test_loss.append(float(m.group(4)))
            elif re.match(pattern2, line):
                m = re.match(pattern2, line)
                fl_test_initial_acc.append(float(m.group(1)))
                fl_test_initial_loss.append(float(m.group(2)))
            elif re.match(pattern3, line):
                m = re.match(pattern3, line)
                fl_test_train_acc.append(float(m.group(1)))
                fl_test_train_loss.append(float(m.group(2)))
            elif re.match(pattern4, line):
                m = re.match(pattern4, line)
                fl_test_test_acc.append(float(m.group(1)))
                fl_test_test_loss.append(float(m.group(2)))

        fl_support_test_acc = fl_support_test_acc[:args.num_rounds + 1]
        fl_test_test_acc = fl_test_test_acc[:args.num_rounds + 1]


    if args.fr_log_path:
        m = re.match('.*B(\d+).*', args.fr_log_path)
        fr_batch_size = m.group(1)
        m = re.match('.*E(\d+).*', args.fr_log_path)
        fr_num_epochs = m.group(1)
        m = re.match('.*C(0.\d+).*', args.fr_log_path)
        fr_fraction = m.group(1)

        fr_support_train_acc = []
        fr_support_train_loss = []
        fr_support_test_acc = []
        fr_support_test_loss = []
        fr_test_initial_acc = []
        fr_test_initial_loss = []
        fr_test_train_acc = []
        fr_test_train_loss = []
        fr_test_test_acc = []
        fr_test_test_loss = []

        reader = open(args.fr_log_path, 'r')
        while True:
            line = reader.readline()
            if len(line) == 0:
                break
            line = line.rstrip()
            if re.match(pattern1, line):
                m = re.match(pattern1, line)
                fr_support_train_acc.append(float(m.group(1)))
                fr_support_train_loss.append(float(m.group(2)))
                fr_support_test_acc.append(float(m.group(3)))
                fr_support_test_loss.append(float(m.group(4)))
            elif re.match(pattern2, line):
                m = re.match(pattern2, line)
                fr_test_initial_acc.append(float(m.group(1)))
                fr_test_initial_loss.append(float(m.group(2)))
            elif re.match(pattern3, line):
                m = re.match(pattern3, line)
                fr_test_train_acc.append(float(m.group(1)))
                fr_test_train_loss.append(float(m.group(2)))
            elif re.match(pattern4, line):
                m = re.match(pattern4, line)
                fr_test_test_acc.append(float(m.group(1)))
                fr_test_test_loss.append(float(m.group(2)))

        fr_support_test_acc = fr_support_test_acc[:args.num_rounds + 1]
        fr_test_test_acc = fr_test_test_acc[:args.num_rounds + 1]


    plt.figure(1)
    plt.title('Test Accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.ylim(bottom=0.0, top=100.0)

    x = np.arange(1, len(fl_support_test_acc) + 1)
    l1 = plt.plot(x, fl_support_test_acc, 'r-', label=('FL B={} E={} C={}'
                                                       .format(fl_batch_size, fl_num_epochs, fl_fraction)))
    plt.plot(x, fl_support_test_acc, 'r-')

    x = np.arange(1, len(fr_support_test_acc) + 1)
    l2 = plt.plot(x, fr_support_test_acc, 'b-', label=('FR B={} E={} C={}'
                                                       .format(fr_batch_size, fr_num_epochs, fr_fraction)))
    plt.plot(x, fr_support_test_acc, 'b-')

    plt.legend()
    plt.show()

#######################################################################################################

    plt.figure(2)
    plt.title('Localization')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.ylim(bottom=0.0, top=100.0)

    x = np.arange(1, len(fl_test_test_acc) + 1)
    l3 = plt.plot(x, fl_test_test_acc, 'r-', label=('FL B={} E={} C={}'
                                                       .format(fl_batch_size, fl_num_epochs, fl_fraction)))
    plt.plot(x, fl_test_test_acc, 'r-')

    x = np.arange(1, len(fr_test_test_acc) + 1)
    l4 = plt.plot(x, fr_test_test_acc, 'b-', label=('FR B={} E={} C={}'
                                                       .format(fr_batch_size, fr_num_epochs, fr_fraction)))
    plt.plot(x, fr_test_test_acc, 'b-')

    plt.legend()
    plt.show()

