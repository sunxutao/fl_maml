import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model
from utils import AccuracyCompute

def run(input_data, model, loss_func, optimizer=None):
    loss_list, acc_list = [], []
    if optimizer: model.train()
    else: model.eval()

    for i, data in enumerate(input_data):
        if optimizer:
            optimizer.zero_grad()
        inputs = data[0]
        labels = data[-1]
        inputs = inputs.type(torch.float32)
        labels = labels.type(torch.long)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss_list.append(loss)
        acc_list.append(AccuracyCompute(outputs, labels))
        if optimizer:
            loss.backward()
            optimizer.step()

    accuracy = np.mean(acc_list, axis=0)
    loss = torch.mean(torch.stack(loss_list))
    return accuracy, loss

def local_train(data_train, data_test, args, initial_weights):
    train_data = Data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_data = Data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    model = create_model(args)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # load weights
    model.load_state_dict(initial_weights)

    # train
    for epoch_idx in range(args.num_epochs):
        run(train_data, model, loss_func, optimizer)

    # eval
    with torch.no_grad():
        train_acc, train_loss = run(train_data, model, loss_func)
        test_acc, test_loss = run(test_data, model, loss_func)

    return model, train_acc, test_acc, train_loss, test_loss

def aggregation(data_train, data_test, args, clientIDs, initial_weights):
    all_train_acc, all_train_loss = [], []
    all_test_acc, all_test_loss = [], []
    model_list = []

    for clientID in clientIDs:
        model, train_acc, test_acc, train_loss, test_loss = local_train(data_train[clientID], data_test[clientID],
                                                                        args, initial_weights)
        model_list.append(model)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

    for key in model_list[0].state_dict().keys():
        for i in range(1, len(model_list)):
            model_list[0].state_dict()[key].add_(model_list[i].state_dict()[key])
        model_list[0].state_dict()[key] /= len(clientIDs)

    mean_train_acc = np.mean(all_train_acc, axis=0)
    mean_train_loss = torch.mean(torch.stack(all_train_loss))
    mean_test_acc = np.mean(all_test_acc, axis=0)
    mean_test_loss = torch.mean(torch.stack(all_test_loss))
    print('average_train_accuracy={}' .format(mean_train_acc))
    print('average_train_loss={}' .format(mean_train_loss))
    print('average_test_accuracy={}' .format(mean_test_acc))
    print('average_test_loss={}' .format(mean_test_loss))
    return model_list[0].state_dict(), mean_train_acc, mean_test_acc, mean_train_loss, mean_test_loss

def FL(data_train, data_test, args):
    train_acc, train_loss = [], []
    test_acc, test_loss = [], []

    # number of selected clients per round
    num_clients = int(args.fraction * len(data_train))

    print('FL:\n')

    # initial model
    model = create_model(args)
    weights = model.state_dict()

    # FL iterations
    for round_num in range(1, args.num_rounds + 1):
        print('round {:2d}:' .format(round_num))
        clientIDs = np.random.choice(range(len(data_train)), num_clients, replace=False)
        weights, acc1, acc2, loss1, loss2 = aggregation(data_train, data_test, args, clientIDs, weights)
        train_acc.append(acc1)
        train_loss.append(loss1)
        test_acc.append(acc2)
        test_loss.append(loss2)
        print()
    return train_acc, test_acc, train_loss, test_loss