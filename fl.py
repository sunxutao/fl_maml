import numpy as np
import logging
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model, AvgrageMeter, client_update
import copy


def aggregation(data_train, data_test, args, clientIDs, model, optimizer):
    mean_train_acc, mean_train_loss = AvgrageMeter(), AvgrageMeter()
    mean_test_acc, mean_test_loss = AvgrageMeter(), AvgrageMeter()

    initial_weights = copy.deepcopy(model.state_dict())

    num_examples = []
    weight_dict_list = []

    for clientID in clientIDs:
        model.load_state_dict(initial_weights)
        model, train_acc, train_loss, test_acc_list, test_loss_list=client_update(data_train[clientID],
                                    data_test[clientID], args, model, optimizer, args.train_epochs)
        num_examples.append(len(data_train[clientID]))
        # load state_dict for each client
        weight_dict_list.append(copy.deepcopy(model.state_dict()))
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



def FL(d_train, d_test, args):
    # Convert to data loader
    for i in range(len(d_train)):
        d_train[i] = Data.DataLoader(d_train[i], batch_size=args.batch_size, shuffle=True)
        d_test[i] = Data.DataLoader(d_test[i], batch_size=args.batch_size, shuffle=False)

    logging.info('number_client: {}'.format(len(d_train)))

    # number of selected clients per round
    num_clients = int(args.fraction * len(d_train))
    logging.info('number_selected_clients_per_round: {}'.format(num_clients))

    print('FL:\n')

    # initial model
    model = create_model(args)

    optimizer = optim.SGD(model.parameters(), lr=args.train_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_rounds)

    # FL iterations
    for round_num in range(1, args.num_rounds + 1):
        # update inner lr
        optimizer = optim.SGD(model.parameters(), lr=scheduler.get_lr()[0])
        scheduler.step()

        # Train on support client sets
        clientIDs = np.random.choice(range(len(d_train)), num_clients, replace=False)
        weights, train_acc, train_loss, acc1, loss1= aggregation(d_train, d_test, args, clientIDs, model, optimizer)
        model.load_state_dict(weights)

        # log info
        logging.info('round {:2d}: train_acc {:.6f}, train_loss {:.6f}, test_acc {:.6f}, '
                     'test_loss {:.6f}'.format(round_num, train_acc, train_loss, acc1, loss1))

    return