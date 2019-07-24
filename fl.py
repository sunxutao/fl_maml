import numpy as np
import logging
import torch
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model, AvgrageMeter, run


def local_train(data_train, data_test, args, initial_weights, num_epochs, lr):
    # create and initialize model
    model = create_model(args, initial_weights)

    # create local optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    test_acc_list, test_loss_list = [], []
    for epoch_idx in range(num_epochs):
        train_acc, train_loss = run(data_train, model, args.loss_func, optimizer)
        with torch.no_grad():
            test_acc, test_loss = run(data_test, model, args.loss_func)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    return model, train_acc, train_loss, test_acc_list, test_loss_list

def aggregation(data_train, data_test, args, clientIDs, initial_weights):
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()
    num_examples = []
    weight_dict_list = []

    for clientID in clientIDs:
        model, train_acc, train_loss, test_acc_list, test_loss_list=local_train(data_train[clientID],
                                    data_test[clientID], args, initial_weights, args.train_epochs, args.train_lr)
        num_examples.append(len(data_train[clientID]))
        # load state_dict for each client
        weight_dict_list.append(model.state_dict())
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        mean_test_acc.update(test_acc_list[-1], 1)
        mean_test_loss.update(test_loss_list[-1], 1)

    # fedAveraging
    for key in weight_dict_list[0].keys():
        weight_dict_list[0][key] *= num_examples[0]
        for model_id in range(1, len(weight_dict_list)):
            weight_dict_list[0][key].add_(weight_dict_list[model_id][key] * num_examples[model_id])
        weight_dict_list[0][key].div_(np.sum(num_examples))

    return weight_dict_list[0], mean_train_acc.avg, mean_train_loss.avg, mean_test_acc.avg, mean_test_loss.avg

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
def localization(data_train, data_test, args, initial_weights):
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    test_acc, test_loss = [], []

    for clientID in range(len(data_train)):
        _, train_acc, train_loss, test_acc_list, test_loss_list=local_train(data_train[clientID], data_test[clientID],
                                                    args, initial_weights, args.local_epochs, args.local_lr)
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        test_acc.append(test_acc_list)
        test_loss.append(test_loss_list)

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)

    return mean_train_acc.avg, mean_train_loss.avg, test_acc, test_loss

def FL(support_train, support_test, test_train, test_test, args):
    # Convert to data loader
    for i in range(len(support_train)):
        support_train[i] = Data.DataLoader(support_train[i], batch_size=args.batch_size, shuffle=True)
        support_test[i] = Data.DataLoader(support_test[i], batch_size=args.batch_size, shuffle=False)
    for i in range(len(test_train)):
        test_train[i] = Data.DataLoader(test_train[i], batch_size=args.batch_size, shuffle=True)
        test_test[i] = Data.DataLoader(test_test[i], batch_size=args.batch_size, shuffle=False)

    # number of selected clients per round
    num_clients = int(args.fraction * len(support_train))

    print('FL:\n')

    # initial model
    model = create_model(args)
    weights = model.state_dict()

    optimizer = optim.SGD(model.parameters(), lr=args.train_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_rounds)

    # FL iterations
    for round_num in range(1, args.num_rounds + 1):
        logging.info('round {:2d}:' .format(round_num))

        # Train on support client sets
        args.train_lr = scheduler.get_lr()[0]
        scheduler.step()
        clientIDs = np.random.choice(range(len(support_train)), num_clients, replace=False)
        weights, train_acc, train_loss, acc1, loss1= aggregation(support_train, support_test, args, clientIDs, weights)

        # log info
        logging.info('support_train_acc {:.6f}, support_train_loss {:.6f},'
                     'support_test_acc {:.6f}, support_test_loss {:.6f}' .format(train_acc, train_loss, acc1, loss1))
        if round_num % args.test_client_interval == 0:
            # update model
            model.load_state_dict(weights)

            # Eval on test client sets with current weights
            acc2, loss2 = evaluation(test_test, args, model)
            # log info
            logging.info('initial_acc {:.6f}, initial_loss {:.6f}'.format(acc2, loss2))

            # Eval on test client sets with localization
            acc3, loss3, test_acc, test_loss = localization(test_train, test_test, args, weights)
            # log info
            logging.info('localization_acc {:.6f}, localization_loss {:.6f}'
                         .format(acc3, loss3))
            for i in range(len(test_acc)):
                logging.info('epoch: {:2d}: test acc: {:.6f}, test loss: {:.6f}' .format(i, test_acc[i], test_loss[i]))

    return