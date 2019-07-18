
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from fl import FL
from maml import MAML


parser = argparse.ArgumentParser("FL_MAML")

# hyper-parameters
parser.add_argument('--fraction', type=float, default=0.01, help='fraction of selected clients')
parser.add_argument('--num_rounds', type=int, default=200, help='number of communication rounds')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')

args = parser.parse_args()

hf_train = h5py.File('fed_emnist_digitsonly_train.h5', 'r')
hf_test = h5py.File('fed_emnist_digitsonly_test.h5', 'r')

data_train = [hf_train['examples'][group]['pixels'][()] for group in hf_train['examples']]
label_train = [hf_train['examples'][group]['label'][()] for group in hf_train['examples']]
data_test = [hf_test['examples'][group]['pixels'][()] for group in hf_test['examples']]
label_test = [hf_test['examples'][group]['label'][()] for group in hf_test['examples']]

# print(sum_train, sum_test) #341873.0 40832.0


lr_alpha = args.learning_rate
#lr_beta = 0.2
num_epochs = args.num_epochs
batch_size = args.batch_size
num_rounds = args.num_rounds
num_clients = int(args.fraction * len(data_train))
fl_train_acc, fl_test_acc, fl_train_loss, fl_test_loss = FL(num_rounds, num_clients, data_train, label_train, data_test,
                                                            label_test, lr_alpha, num_epochs, batch_size)
# maml_train_acc, maml_test_acc = MAML(num_rounds, num_clients, data_train, label_train, data_test, label_test, lr_beta,
#                                      lr_alpha, num_epochs, batch_size)

plt.figure(1)
x=np.arange(1,num_rounds + 1)
l1=plt.plot(x,fl_train_acc,'r-',label='FL_train')
l2=plt.plot(x,fl_test_acc,'r--',label='FL_test')
#l3=plt.plot(x,maml_train_acc,'b-',label='MAML_train')
#l4=plt.plot(x,maml_test_acc,'b--',label='MAML_test')
plt.plot(x,fl_train_acc,'r-',x,fl_test_acc,'r--')
#plt.plot(x,maml_train_acc,'bo-',x,maml_test_acc,'bo--')
plt.title('FL vs MAML')
plt.xlabel('round')
plt.ylabel('accuracy')
plt.ylim(bottom=0.0,top=1.0)
#plt.rcParams['figure.dpi'] = 500
plt.legend()
plt.show()

plt.figure(2)
x=np.arange(1,num_rounds + 1)
l1=plt.plot(x,fl_train_loss,'b-',label='FL_train')
l2=plt.plot(x,fl_test_loss,'b--',label='FL_test')
#l3=plt.plot(x,maml_train_acc,'b-',label='MAML_train')
#l4=plt.plot(x,maml_test_acc,'b--',label='MAML_test')
plt.plot(x,fl_train_loss,'b-',x,fl_test_loss,'b--')
#plt.plot(x,maml_train_acc,'bo-',x,maml_test_acc,'bo--')
plt.title('FL vs MAML')
plt.xlabel('round')
plt.ylabel('loss')

#plt.rcParams['figure.dpi'] = 500
plt.legend()
plt.show()