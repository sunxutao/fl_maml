import numpy as np
import matplotlib.pyplot as plt
import h5py
from fl import FL
from maml import MAML

hf_train = h5py.File('fed_emnist_digitsonly_train.h5', 'r')
hf_test = h5py.File('fed_emnist_digitsonly_test.h5', 'r')

data_train = [hf_train['examples'][group]['pixels'][()] for group in hf_train['examples']]
label_train = [hf_train['examples'][group]['label'][()] for group in hf_train['examples']]
data_test = [hf_test['examples'][group]['pixels'][()] for group in hf_test['examples']]
label_test = [hf_test['examples'][group]['label'][()] for group in hf_test['examples']]

lr_alpha = 0.1
lr_beta = 0.5
num_epochs = 20
batch_size = 128
num_rounds = 100
num_clients = 2
fl_train_acc, fl_test_acc = FL(num_rounds, num_clients, data_train, label_train, data_test, label_test, lr_alpha, num_epochs, batch_size)
maml_train_acc, maml_test_acc = MAML(num_rounds, num_clients, data_train, label_train, data_test, label_test, lr_beta, lr_alpha, num_epochs, batch_size)

x=np.arange(1,num_rounds + 1)
l1=plt.plot(x,fl_train_acc,'r-',label='FL_train')
l2=plt.plot(x,fl_test_acc,'r--',label='FL_test')
l3=plt.plot(x,maml_train_acc,'b-',label='MAML_train')
l4=plt.plot(x,maml_test_acc,'b--',label='MAML_test')
plt.plot(x,fl_train_acc,'ro-',x,fl_test_acc,'ro--')
plt.plot(x,maml_train_acc,'bo-',x,maml_test_acc,'bo--')
plt.title('FL vs MAML')
plt.xlabel('round')
plt.ylabel('accuracy')
plt.ylim(bottom=0.0,top=1.0)
#plt.rcParams['figure.dpi'] = 500
plt.legend()
plt.show()