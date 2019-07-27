import h5py
import torch
import os
import numpy as np
import shutil
import myModel
import torch.optim as optim
import time
import copy

def load_data(hdf5_file, args):
    file = h5py.File(hdf5_file, 'r')
    if args.model == 'LSTM':
        data = [file['examples'][group]['snippets'][()] for group in file['examples']]
        num_clients = len(data)
        processed_data = []
        for i in range(num_clients):
            whole_string = ''
            for j in range(len(data[i])):
                whole_string += data[i][j].decode('utf-8')
            processed_data.append(whole_string.lower())
    else:
        data = [file['examples'][group]['pixels'][()] for group in file['examples']]
        label = [file['examples'][group]['label'][()] for group in file['examples']]
        num_clients = len(data)
        processed_data = []
        for i in range(num_clients):
            if (args.model == 'MLP'):
                data[i]=torch.tensor(data[i].reshape(data[i].shape[0], 784), device=args.device, dtype=torch.float32)
            elif (args.model == 'LeNet'):
                data[i]=torch.tensor(data[i].reshape(data[i].shape[0], 1, 28, 28), device=args.device, dtype=torch.float32)
            label[i]=torch.tensor(label[i], device=args.device, dtype=torch.long)
            processed_data.append(list(zip(data[i], label[i])))
    return processed_data

def split_data(train, test, args):
    num_support = int(len(train) * args.fraction_t)
    support_IDs = np.random.choice(range(len(train)), num_support, replace=False)
    support_train, support_test, test_train, test_test = [], [], [], []
    for i in range(len(train)):
        if i in support_IDs:
            support_train.append(train[i])
            support_test.append(test[i])
        else:
            test_train.append(train[i])
            test_test.append(test[i])
    return support_train, support_test, test_train, test_test


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if not os.path.exists(script_path):
            os.mkdir(script_path)
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)



def create_model(args, initial_weights=None):
    if args.model == 'MLP':
        model = myModel.MLP().to(args.device)
    elif args.model == 'LeNet':
        model = myModel.LeNet().to(args.device)
    elif args.model == 'LSTM':
        model = myModel.LSTM(args.ntokens).to(args.device)
    if initial_weights: model.load_state_dict(initial_weights)
    return model

def run(input_data, model, loss_func, optimizer=None):
    loss_avg, acc_avg = AvgrageMeter(), AvgrageMeter()
    if optimizer:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(input_data):
        if optimizer:
            optimizer.zero_grad()
        inputs = data[0]
        labels = data[-1]
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        acc = accuracy(outputs, labels)[0]

        loss_avg.update(loss.data.item(), inputs.size(0))
        acc_avg.update(acc.data.item(), inputs.size(0))
        if optimizer:
            loss.backward()
            optimizer.step()
    return acc_avg.avg, loss_avg.avg

def client_update(data_train, data_test, args, model, optimizer, num_epochs):
    test_acc_list, test_loss_list = [], []
    for epoch_idx in range(num_epochs):
        train_acc, train_loss = run(data_train, model, args.loss_func, optimizer)
        with torch.no_grad():
            test_acc, test_loss = run(data_test, model, args.loss_func)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    return model, train_acc, train_loss, test_acc_list, test_loss_list

# eval model on data
def evaluation(data, args, model):
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()
    for clientID in range(len(data)):
        with torch.no_grad():
            test_acc, test_loss = run(data[clientID], model, args.loss_func)
        mean_test_acc.update(test_acc, 1)
        mean_test_loss.update(test_loss, 1)

    return mean_test_acc.avg, mean_test_loss.avg

# model localization
def localization(data_train, data_test, args, model, optimizer):
    # save the model weights
    initial_weights = copy.deepcopy(model.state_dict())
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    test_acc, test_loss = [], []

    for clientID in range(len(data_train)):
        model.load_state_dict(initial_weights)
        _, train_acc, train_loss, test_acc_list, test_loss_list=client_update(data_train[clientID], data_test[clientID],
                                                    args, model, optimizer, args.local_epochs)
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        test_acc.append(test_acc_list)
        test_loss.append(test_loss_list)

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)
    # load the model weights
    model.load_state_dict(initial_weights)
    return mean_train_acc.avg, mean_train_loss.avg, test_acc, test_loss

###########################################################################################
# LSTM

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def tokenize(client_text, dictionary):
    """Tokenizes a text file."""
    # Add words to the dictionary
    tokens = 0
    words = client_text.split() + ['<eos>']
    tokens += len(words)
    for word in words:
        dictionary.add_word(word)

    # Tokenize file content
    ids = torch.LongTensor(tokens)
    token = 0
    words = client_text.split() + ['<eos>']
    for word in words:
        ids[token] = dictionary.word2idx[word]
        token += 1
    return ids


def batchify(data, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // args.batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * args.batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(args.batch_size, -1).t().contiguous()
    return data.to(args.device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def lstm_data_process(dtrain, dtest, corpus, args):
    d_train = []  # train data of support client after processing
    d_test = []  # test data of support client after processing
    for i in range(len(dtrain)):
        d_train.append(batchify(tokenize(dtrain[i], corpus), args))
        d_test.append(batchify(tokenize(dtest[i], corpus), args))
    return d_train, d_test


def lstm_run(input_data, model, args, optimizer=None):
    loss_avg, acc_avg = AvgrageMeter(), AvgrageMeter()
    if optimizer:
        model.train()
    else:
        model.eval()
    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, input_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(args, input_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        if optimizer:
            optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = args.loss_func(output.view(-1, args.ntokens), targets)
        acc = accuracy(output.view(-1, args.ntokens), targets)[0]
        loss_avg.update(loss.data.item(), data.size(0))
        acc_avg.update(acc.data.item(), data.size(0))
        if optimizer:
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-args.lstm_lr, p.grad.data)

    return acc_avg.avg, loss_avg.avg

def lstm_train(data_train, data_test, args, model, optimizer, num_epochs):
    # model = create_model(args, initial_weights)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    test_acc_list, test_loss_list = [], []
    for epoch_idx in range(num_epochs):
        train_acc, train_loss = lstm_run(data_train, model, args, optimizer)
        with torch.no_grad():
            test_acc, test_loss = lstm_run(data_test, model, args)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        # print('epoch: {} train_acc: {:.4f} train_loss: {:.4f} test_acc: {:.4f} test_loss: {:.4f}'
        #       .format(epoch_idx + 1, train_acc, train_loss, test_acc, test_loss))

    return model, train_acc, train_loss, test_acc_list, test_loss_list

def lstm_evaluation(data, args, model):
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()
    for clientID in range(len(data)):
        with torch.no_grad():
            test_acc, test_loss = lstm_run(data[clientID], model, args)
        mean_test_acc.update(test_acc, 1)
        mean_test_loss.update(test_loss, 1)

    return mean_test_acc.avg, mean_test_loss.avg

# model localization
def lstm_localization(data_train, data_test, args, model, optimizer):
    # save the model weights
    initial_weights = copy.deepcopy(model.state_dict())
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    test_acc, test_loss = [], []

    for clientID in range(len(data_train)):
        model.load_state_dict(initial_weights)
        _, train_acc, train_loss, test_acc_list, test_loss_list=lstm_train(data_train[clientID], data_test[clientID],
                                                    args, model, optimizer, args.local_epochs)
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        test_acc.append(test_acc_list)
        test_loss.append(test_loss_list)

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)
    # load the model weights
    model.load_state_dict(initial_weights)
    return mean_train_acc.avg, mean_train_loss.avg, test_acc, test_loss