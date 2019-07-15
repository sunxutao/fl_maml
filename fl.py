import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import myModel
from myModel import AccuracyCompute

def local_train(x_train, y_train, x_test, y_test, lr, num_epochs, batch_size, weights=None):
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    train_data = np.column_stack((x_train,y_train))
    train_data = Data.DataLoader(train_data,batch_size=batch_size)

    mlp = myModel.MLP()

    optimizer = optim.SGD(mlp.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # load weights
    if weights is not None:
        mlp.load_state_dict(weights)

    mlp.train()

    # train
    for epoch_idx in range(num_epochs):
        for i, data in enumerate(train_data):
            optimizer.zero_grad()
            inputs = data[:, 0:-1]
            labels = data[:, -1]
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = mlp(inputs.float())
            loss = loss_func(outputs,labels.long())
            loss.backward()
            optimizer.step()

    mlp.eval()

    inputs = Variable(torch.tensor(x_train))
    labels = Variable(torch.tensor(y_train))
    outputs = mlp(inputs.float())
    train_accuracy = AccuracyCompute(outputs, labels.long())

    test_inputs = Variable(torch.tensor(x_test))
    test_labels = Variable(torch.tensor(y_test))
    test_outputs = mlp(test_inputs.float())
    test_accuracy = AccuracyCompute(test_outputs, test_labels.long())

    return mlp.named_parameters(), train_accuracy, test_accuracy

def aggregation(num_clients, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size, initial_weights=None):
    mean_weights = {}
    all_train_acc = []
    all_test_acc = []
    for clientID in range(1, num_clients + 1):
        weights, train_acc, test_acc = local_train(x_train[clientID], y_train[clientID], x_test[clientID], y_test[clientID], lr, num_epochs, batch_size, initial_weights)
        weights = dict(weights)
        if clientID == 1:
            for index,key in enumerate(weights):
                mean_weights[key] = weights[key]
        else:
            for index,key in enumerate(weights):
                mean_weights[key] = weights[key] + mean_weights[key]
        all_train_acc.append(train_acc)
        all_test_acc.append(test_acc)
        # print('client {:2d}:\n train accuracy={}\n test accuracy={}'
        #       .format(clientID, train_acc, test_acc))
    for key in mean_weights.keys():
        mean_weights[key] = mean_weights[key] / num_clients
    mean_train_acc = np.mean(all_train_acc, axis=0)
    mean_test_acc = np.mean(all_test_acc, axis=0)
    print('average_train_accuracy={}' .format(mean_train_acc))
    print('average_test_accuracy={}' .format(mean_test_acc))
    return mean_weights, mean_train_acc, mean_test_acc

def FL(num_rounds, num_clients, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size):
    train_acc=[]
    test_acc=[]
    print('FL:\n')
    # initial model
    print('initialization round:')
    weights = aggregation(num_clients, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size)[0]
    print()
    # FL iterations
    for round_num in range(1, num_rounds + 1):
        print('round {:2d}:' .format(round_num))
        weights, acc1, acc2 = aggregation(num_clients, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size, weights)
        train_acc.append(acc1)
        test_acc.append(acc2)
        print()
    return train_acc, test_acc