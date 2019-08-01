import numpy as np
import logging
import torch
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model, AvgrageMeter, client_update, evaluation, localization, run
import copy


def aggregation(data_train, data_test, args, clientIDs, model, optimizer):
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()

    initial_weights = copy.deepcopy(model.state_dict())

    weight_dict_list = []

    for clientID in clientIDs:
        model.load_state_dict(initial_weights)
        model, train_acc, train_loss, test_acc_list, test_loss_list=client_update(data_train[clientID],
                                    data_test[clientID], args, model, optimizer, 1)
        # load state_dict for each client
        weight_dict_list.append(copy.deepcopy(model.state_dict()))
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


def FR(dsupport_train, dsupport_test, dtest_train, dtest_test, args):
    # filter unqualified client sets
    support_train = [dsupport_train[i] for i in range(len(dsupport_train))
                     if len(dsupport_train[i]) > 30 and len(dsupport_train[i]) < 500]
    support_test = [dsupport_test[i] for i in range(len(support_train))
                     if len(support_train[i]) > 30 and len(support_train[i]) < 500]
    test_train = [dtest_train[i] for i in range(len(dtest_train))
                     if len(dtest_train[i]) > 30 and len(dtest_train[i]) < 500]
    test_test = [dtest_test[i] for i in range(len(test_train))
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

    logging.info('number_support_client: {}'.format(len(support_train)))
    logging.info('number_test_client: {}'.format(len(test_train)))

    # number of selected clients per round
    num_clients = int(args.fraction * len(support_train))
    logging.info('number_selected_clients_per_round: {}'.format(num_clients))

    print('FR:\n')

    # initial model
    model = create_model(args)

    op_inner = optim.SGD(model.parameters(), lr=args.train_lr)
    scheduler_inner = optim.lr_scheduler.CosineAnnealingLR(op_inner, T_max=args.num_rounds)
    op_outer = optim.SGD(model.parameters(), lr=args.global_lr)
    scheduler_outer = optim.lr_scheduler.CosineAnnealingLR(op_outer, T_max=args.num_rounds)
    op_local = optim.SGD(model.parameters(), lr=args.local_lr)

    # FR iterations
    for round_num in range(1, args.num_rounds + 1):
        # update inner lr
        optimizer = optim.SGD(model.parameters(), lr=scheduler_inner.get_lr()[0])
        scheduler_inner.step()
        # update outer lr
        args.global_lr = scheduler_outer.get_lr()[0]
        scheduler_outer.step()

        # Train on support client sets
        clientIDs = np.random.choice(range(len(support_train)), num_clients, replace=False)
        weights, train_acc, train_loss, acc1, loss1=aggregation(support_train, support_test, args, clientIDs, model, optimizer)
        model.load_state_dict(weights)

        # log info
        logging.info('round {:2d}: support_train_acc {:.6f}, support_train_loss {:.6f}, support_test_acc {:.6f}, '
                     'support_test_loss {:.6f}' .format(round_num, train_acc, train_loss, acc1, loss1))

        if round_num % (args.local_interval * args.train_epochs) == 0:
            # Eval on test client sets with current weights
            acc2, loss2 = evaluation(test_test, args, model)
            # log info
            logging.info('initial_acc {:.6f}, initial_loss {:.6f}' .format(acc2, loss2))

            # Eval on test client sets with localization
            acc3, loss3, test_acc, test_loss = localization(test_train, test_test, args, model, op_local)
            #log info
            logging.info('localization_acc {:.6f}, localization_loss {:.6f}' .format(acc3, loss3))
            for i in range(len(test_acc)):
                logging.info('epoch: {:2d}: test acc: {:.6f}, test loss: {:.6f}'.format(i+1, test_acc[i], test_loss[i]))

    return