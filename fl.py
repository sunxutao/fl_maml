import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import myModel
from myModel import AccuracyCompute

def train(input_data, model, loss_func, optimizer=None, LeNet=False):
    loss_list, acc_list = [], []

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(input_data):
        if optimizer is not None:
            optimizer.zero_grad()
        inputs = data[:, 0:-1]
        if LeNet == True:
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
        labels = data[:, -1]
        outputs = model(inputs.float())
        loss = loss_func(outputs, labels.long())
        loss_list.append(loss)
        acc_list.append(AccuracyCompute(outputs, labels.long()))
        if optimizer is not None:
            loss.backward()
            optimizer.step()

    accuracy = np.mean(acc_list, axis=0)
    loss = torch.mean(torch.stack(loss_list))
    return accuracy, loss

def local_train(x_train, y_train, x_test, y_test, lr, num_epochs, batch_size, weights=None):
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    train_data = np.column_stack((x_train,y_train))
    train_data = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = np.column_stack((x_test, y_test))
    test_data = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = myModel.MLP()
    #model = myModel.LeNet()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # load weights
    if weights is not None:
        model.load_state_dict(weights)

    # train
    for epoch_idx in range(num_epochs):
        train(train_data, model, loss_func, optimizer)

    train_acc, train_loss = train(train_data, model, loss_func)
    test_acc, test_loss = train(test_data, model, loss_func)

    return model, train_acc, test_acc, train_loss, test_loss

def aggregation(num_clients,client_indexes,x_train,y_train,x_test,y_test,lr,num_epochs,batch_size,initial_weights=None):
    all_train_acc, all_train_loss = [], []
    all_test_acc, all_test_loss = [], []
    model_list = []

    for clientID in client_indexes:
        model, train_acc, test_acc, train_loss, test_loss = local_train(x_train[clientID], y_train[clientID],
                                                                        x_test[clientID], y_test[clientID], lr,
                                                                        num_epochs, batch_size, initial_weights)
        model_list.append(model)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

    for key in model_list[0].state_dict().keys():
        for i in range(1, len(model_list)):
            model_list[0].state_dict()[key].add_(model_list[i].state_dict()[key])
        model_list[0].state_dict()[key] /= num_clients

    mean_train_acc = np.mean(all_train_acc, axis=0)
    mean_train_loss = torch.mean(torch.stack(all_train_loss))
    mean_test_acc = np.mean(all_test_acc, axis=0)
    mean_test_loss = torch.mean(torch.stack(all_test_loss))
    print('average_train_accuracy={}' .format(mean_train_acc))
    print('average_train_loss={}' .format(mean_train_loss))
    print('average_test_accuracy={}' .format(mean_test_acc))
    print('average_test_loss={}' .format(mean_test_loss))
    return model_list[0].state_dict(), mean_train_acc, mean_test_acc, mean_train_loss, mean_test_loss

def FL(num_rounds, num_clients, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size):
    train_acc, train_loss = [], []
    test_acc, test_loss = [], []
    print('FL:\n')
    # initial model
    print('initialization round:')
    initial_indexes = np.random.choice(range(len(x_train)), num_clients, replace=False)
    weights = aggregation(num_clients, initial_indexes, x_train, y_train, x_test, y_test, lr, num_epochs, batch_size)[0]
    print()
    # FL iterations
    for round_num in range(1, num_rounds + 1):
        print('round {:2d}:' .format(round_num))
        indexes_per_round = np.random.choice(range(len(x_train)), num_clients, replace=False)
        weights, acc1, acc2, loss1, loss2 = aggregation(num_clients, indexes_per_round, x_train, y_train, x_test, y_test,
                                                        lr, num_epochs, batch_size, weights)
        train_acc.append(acc1)
        train_loss.append(loss1)
        test_acc.append(acc2)
        test_loss.append(loss2)
        print()
    return train_acc, test_acc, train_loss, test_loss