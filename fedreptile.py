import numpy as np
import logging
import torch
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model, AvgrageMeter, run


def client_update(data_train, data_test, args, initial_weights, lr, num_epochs=1):
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
    weight_dict_list = []

    for clientID in clientIDs:
        model, train_acc, train_loss, test_acc_list, test_loss_list=client_update(data_train[clientID],
                                    data_test[clientID], args, initial_weights, args.train_lr, 1)
        # load state_dict for each client
        weight_dict_list.append(model.state_dict())
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        mean_test_acc.update(test_acc_list[-1], 1)
        mean_test_loss.update(test_loss_list[-1], 1)

    # meta-learning
    for key in weight_dict_list[0].keys():
        for model_id in range(1, len(weight_dict_list)):
            weight_dict_list[0][key].add_(weight_dict_list[model_id][key])
        weight_dict_list[0][key].mul_(args.global_lr/len(clientIDs)).add_((1-args.global_lr)*initial_weights[key])

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
        _, train_acc, train_loss, test_acc_list, test_loss_list=client_update(data_train[clientID], data_test[clientID],
                                                    args, initial_weights, args.local_lr, args.local_epochs)
        mean_train_acc.update(train_acc, 1)
        mean_train_loss.update(train_loss, 1)
        test_acc.append(test_acc_list)
        test_loss.append(test_loss_list)

    test_acc = np.mean(test_acc, axis=0)
    test_loss = np.mean(test_loss, axis=0)

    return mean_train_acc.avg, mean_train_loss.avg, test_acc, test_loss

def FR(support_train, support_test, test_train, test_test, args):
    # filter unqualified client sets
    support_train = [support_train[i] for i in range(len(support_train))
                     if len(support_train[i]) > 30 and len(support_train[i]) < 500]
    support_test = [support_test[i] for i in range(len(support_train))
                     if len(support_train[i]) > 30 and len(support_train[i]) < 500]
    test_train = [test_train[i] for i in range(len(test_train))
                     if len(test_train[i]) > 30 and len(test_train[i]) < 500]
    test_test = [test_test[i] for i in range(len(test_train))
                  if len(test_train[i]) > 30 and len(test_train[i]) < 500]

    # convert to data loader and split to batches
    for i in range(len(support_train)):
        batch_size = int(len(support_train[i]) / args.inner_iterations)
        support_train[i] = Data.DataLoader(support_train[i], batch_size=batch_size, shuffle=True, drop_last=True)
        support_test[i] = Data.DataLoader(support_test[i], batch_size=args.batch_size, shuffle=False)
    for i in range(len(test_train)):
        batch_size = int(len(test_train[i]) / args.local_epochs)
        test_train[i] = Data.DataLoader(test_train[i], batch_size=batch_size, shuffle=True, drop_last=True)
        test_test[i] = Data.DataLoader(test_test[i], batch_size=args.batch_size, shuffle=False)

    # number of selected clients per round
    num_clients = int(args.fraction * len(support_train))

    print('FR:\n')

    # initial model
    model = create_model(args)
    weights = model.state_dict()

    op_inner = optim.SGD(model.parameters(), lr=args.train_lr)
    op_outer = optim.SGD(model.parameters(), lr=args.global_lr)
    scheduler_inner = optim.lr_scheduler.CosineAnnealingLR(op_inner, T_max=args.num_rounds)
    scheduler_outer = optim.lr_scheduler.CosineAnnealingLR(op_outer, T_max=args.num_rounds)
    # FL iterations
    for round_num in range(1, args.num_rounds + 1):
        logging.info('round {:2d}:' .format(round_num))

        # Train on support client sets
        args.train_lr = scheduler_inner.get_lr()[0]
        scheduler_inner.step()
        args.global_lr = scheduler_outer.get_lr()[0]
        scheduler_outer.step()
        clientIDs = np.random.choice(range(len(support_train)), num_clients, replace=False)
        weights, train_acc, train_loss, acc1, loss1=aggregation(support_train, support_test, args, clientIDs, weights)
        # log info
        logging.info('support_train_acc {:.6f}, support_train_loss {:.6f},'
                     'support_test_acc {:.6f}, support_test_loss {:.6f}' .format(train_acc, train_loss, acc1, loss1))

        if round_num % args.test_client_interval == 0:
            # update model
            model.load_state_dict(weights)

            # Eval on test client sets with current weights
            acc2, loss2 = evaluation(test_test, args, model)
            # log info
            logging.info('initial_acc {:.6f}, initial_loss {:.6f}' .format(acc2, loss2))

            # Eval on test client sets with localization
            acc3, loss3, test_acc, test_loss = localization(test_train, test_test, args, weights)
            #log info
            logging.info('localization_acc {:.6f}, localization_loss {:.6f}'
                         .format(acc3, loss3))
            for i in range(len(test_acc)):
                logging.info('epoch: {:2d}: test acc: {:.6f}, test loss: {:.6f}' .format(i, test_acc[i], test_loss[i]))

    return