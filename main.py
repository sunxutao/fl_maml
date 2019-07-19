import numpy as np
import matplotlib.pyplot as plt
import argparse
from fl import FL
from utils import load_data
#from maml import MAML


parser = argparse.ArgumentParser("FL_MAML")

# hyper-parameters
parser.add_argument('--fraction', type=float, default=0.005, help='fraction of selected clients') # 0.01 0.005
parser.add_argument('--num_rounds', type=int, default=100, help='number of communication rounds')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs') # 1 10 20
parser.add_argument('--batch_size', type=int, default=10, help='batch size') # 10 50
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--model', type=str, default='MLP', help='model') # MLP or LeNet

args = parser.parse_args()

# load data (client: 3383 / train: 341873 / test: 40832)
d_train = load_data('fed_emnist_digitsonly_train.h5', args)
d_test = load_data('fed_emnist_digitsonly_test.h5', args)

fl_train_acc, fl_test_acc, fl_train_loss, fl_test_loss = FL(d_train, d_test, args)
# maml_train_acc, maml_test_acc = MAML(num_rounds, num_clients, data_train, label_train, data_test, label_test, lr_beta,
#                                      lr_alpha, num_epochs, batch_size)

# plt.figure(1)
# x=np.arange(1,num_rounds + 1)
# #l1=plt.plot(x,fl_train_acc,'r-',label=('FL_train B={} E={}' .format(batch_size, num_epochs)))
# l2=plt.plot(x,fl_test_acc,'r--',label=('FL_test B={} E={}' .format(batch_size, num_epochs)))
# #l3=plt.plot(x,maml_train_acc,'b-',label='MAML_train')
# #l4=plt.plot(x,maml_test_acc,'b--',label='MAML_test')
# plt.plot(x,fl_test_acc,'r--')
# #plt.plot(x,maml_train_acc,'bo-',x,maml_test_acc,'bo--')
# plt.title('FL vs MAML')
# plt.xlabel('round')
# plt.ylabel('accuracy')
# plt.ylim(bottom=0.0,top=1.0)
# #plt.rcParams['figure.dpi'] = 500
# plt.legend()
# plt.show()
#
# plt.figure(2)
# x=np.arange(1,num_rounds + 1)
# #l1=plt.plot(x,fl_train_loss,'b-',label='FL_train')
# l2=plt.plot(x,fl_test_loss,'b--',label=('FL_test B={} E={}' .format(batch_size, num_epochs)))
# #l3=plt.plot(x,maml_train_acc,'b-',label='MAML_train')
# #l4=plt.plot(x,maml_test_acc,'b--',label='MAML_test')
# plt.plot(x,fl_test_loss,'b--')
# #plt.plot(x,maml_train_acc,'bo-',x,maml_test_acc,'bo--')
# plt.title('FL vs MAML')
# plt.xlabel('round')
# plt.ylabel('loss')
#
# #plt.rcParams['figure.dpi'] = 500
# plt.legend()
# plt.show()