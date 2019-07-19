import h5py
import numpy as np
import myModel

def load_data(hdf5_file, args):
    file = h5py.File(hdf5_file, 'r')
    data = [file['examples'][group]['pixels'][()] for group in file['examples']]
    label = [file['examples'][group]['label'][()] for group in file['examples']]
    num_clients = len(data)
    processed_data = []
    for i in range(num_clients):
        if (args.model == 'MLP'):
            data[i] = data[i].reshape(data[i].shape[0], 784)
        processed_data.append(list(zip(data[i], label[i])))
    return processed_data

def create_model(args):
    if args.model == 'MLP':
        model = myModel.MLP()
    elif args.model == 'LeNet':
        model = myModel.LeNet()
    return model

def AccuracyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

