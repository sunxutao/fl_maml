import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
import sys
from utils import create_model, AvgrageMeter, run

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
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()
    model_list = []

    for clientID in clientIDs:
        model, train_acc, test_acc, train_loss, test_loss = local_train(data_train[clientID], data_test[clientID],
                                                                        args, initial_weights)
        model_list.append(model)
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        mean_test_acc.update(test_acc, 1)
        mean_test_loss.update(test_loss, 1)

    for key in model_list[0].state_dict().keys():
        for i in range(1, len(model_list)):
            model_list[0].state_dict()[key].add_(model_list[i].state_dict()[key])
        model_list[0].state_dict()[key] /= len(clientIDs)

    logging.info('train_acc {:.6f}, train_loss {:.6f}, test_acc {:.6f}, test_loss {:.6f}'
                 .format(mean_train_acc.avg, mean_train_loss.avg, mean_test_acc.avg, mean_test_loss.avg))
    return model_list[0].state_dict(), mean_train_acc.avg, mean_test_acc.avg, mean_train_loss.avg, mean_test_loss.avg

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