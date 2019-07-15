import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Linear(784, 256, bias=False)
        self.hidden2 = nn.Linear(256, 256, bias=False)
        self.output = nn.Linear(256, 10, bias=False)

    def forward(self, din):
        din = din.view(-1, 28*28)
        dout = nn.functional.relu(self.hidden1(din))
        dout = nn.functional.relu(self.hidden2(dout))
        return nn.functional.softmax(self.output(dout), dim=1)

def AccuracyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

