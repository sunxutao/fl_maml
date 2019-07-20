import h5py
import torch
import os
import shutil
import myModel

def load_data(hdf5_file, args):
    file = h5py.File(hdf5_file, 'r')
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


def create_model(args):
    if args.model == 'MLP':
        model = myModel.MLP().to(args.device)
    elif args.model == 'LeNet':
        model = myModel.LeNet().to(args.device)
    return model

def run(input_data, model, loss_func, optimizer=None):
    loss_avg, acc_avg = AvgrageMeter(), AvgrageMeter()
    if optimizer: model.train()
    else: model.eval()

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