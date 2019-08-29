import numpy as np
import logging
import torch
import torch.optim as optim
import torch.utils.data as Data
from utils import create_model, AvgrageMeter, client_update
import copy


def calculate_distance(weight1, weight2):
    distance = 0
    for key in weight1.keys():
        distance += torch.sum((weight1[key] - weight2[key]) ** 2)
    return distance

def update_selection_prob(selection_prob, clientIDs, aggregated_update, all_updates):
    # step1: calculate the distances between aggregated_update and all_updates
    distance_list = []
    for i in range(len(all_updates)):
        distance = calculate_distance(aggregated_update, all_updates[i])
        distance_list.append(distance)

    # step2: prob_sum = sum up the probabilities of clientIDs in the current selection_prob
    prob_sum = 0
    for index in clientIDs:
        prob_sum += selection_prob[index]

    # step3: dis_sum = sum up the 1/distance in step1
    distance_list = np.reciprocal(distance_list)
    dis_sum = sum(distance_list)

    # step4: selection_prob = update the probability in selection_prob by selection_prob[clientIds[i]] = prob_sum * ((1/dis_i)/dis_sum)
    for i in range(len(all_updates)):
        selection_prob[clientIDs[i]] = prob_sum * (distance_list[i] / dis_sum)
    return selection_prob


def aggregation(data_train, data_test, args, clientIDs, model, optimizer, selection_prob):
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
    if args.robust_method == 'none':
        for key in weight_dict_list[0].keys():
            weight_dict_list[0][key] *= num_examples[0]
            for model_id in range(1, len(weight_dict_list)):
                weight_dict_list[0][key].add_(weight_dict_list[model_id][key] * num_examples[model_id])
            weight_dict_list[0][key].div_(np.sum(num_examples))
        aggregated_weight = weight_dict_list[0]
    elif args.robust_method == 'krum' or args.robust_method == 'incentive':  # krum method
        # step1: for each update calculate the distances with the other updates
        k = int(len(clientIDs) - len(clientIDs) * args.poisoning_fraction)
        distance_list = [] # record the distance sum of each client
        for i in range(len(weight_dict_list)):
            distance = []
            for j in range(len(weight_dict_list)):
                if i == j:
                    continue
                difference = calculate_distance(weight_dict_list[i], weight_dict_list[j])
                distance.append(difference)
            # step2: sort the distances of each update, sum up the former k-1 distances
            distance.sort()
            distance = distance[:k-1]
            sum_distance = sum(distance)
            distance_list.append(sum_distance)

        # step3: find the update (e.g., min_update) with minimum sum
        min_update_index = distance_list.index(min(distance_list))

        # step4: the aggregated update is the average value of min_update and its former k-1 updates
        neighbor_list = []
        for i in range(len(weight_dict_list)):
            if (i == min_update_index):
                continue
            info = []
            difference = 0
            for key in weight_dict_list[i].keys():
                difference += torch.sum((weight_dict_list[i][key] - weight_dict_list[min_update_index][key]) ** 2)
            info.append(difference)
            info.append(i)
            neighbor_list.append(info)
        neighbor_list.sort()
        neighbor_list = neighbor_list[:k-1]
        neighbor_index = []
        for neighbor_info in neighbor_list:
            neighbor_index.append(neighbor_info[1])

        aggregated_weight = copy.deepcopy(weight_dict_list[min_update_index])

        for key in aggregated_weight.keys():
            sum_example = 0
            aggregated_weight[key] *= num_examples[min_update_index]
            sum_example += num_examples[min_update_index]
            for model_id in neighbor_index:
                aggregated_weight[key].add_(weight_dict_list[model_id][key] * num_examples[model_id])
                sum_example += num_examples[model_id]
            aggregated_weight[key].div_(sum_example)

        if args.robust_method == 'incentive':
            # step5: update selection_prob
            selection_prob = update_selection_prob(selection_prob, clientIDs, aggregated_weight, weight_dict_list)

    return aggregated_weight, mean_train_acc.avg, mean_train_loss.avg, mean_test_acc.avg, mean_test_loss.avg, selection_prob

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

    # init a random selection probability array, update in aggregation function when method = 'incentive'
    selection_prob = np.empty(len(d_train))
    selection_prob.fill(1/len(d_train))

    # FL iterations
    for round_num in range(1, args.num_rounds + 1):
        # update inner lr
        optimizer = optim.SGD(model.parameters(), lr=scheduler.get_lr()[0])
        scheduler.step()

        # Train on support client sets
        clientIDs = np.random.choice(range(len(d_train)), num_clients, replace=False, p=selection_prob)
        weights, train_acc, train_loss, acc1, loss1, selection_prob = aggregation(d_train, d_test, args, clientIDs, model, optimizer, selection_prob)
        model.load_state_dict(weights)

        # log info
        logging.info('round {:2d}: train_acc {:.6f}, train_loss {:.6f}, test_acc {:.6f}, '
                     'test_loss {:.6f}'.format(round_num, train_acc, train_loss, acc1, loss1))

    return acc1