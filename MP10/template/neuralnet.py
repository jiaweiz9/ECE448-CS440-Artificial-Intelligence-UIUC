# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader, TensorDataset
from utils import get_parameter_counts


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.lr = lrate
        self.optimizer = optim.Adam(self.parameters(), self.lr)
        # raise NotImplementedError("You need to write this part!")
    
    def l2_regular(self, alpha):
        loss = 0
        for module in self.modules():
            if type(module) is nn.Conv2d:
                loss += (module.weight ** 2).sum() / 2.0
        return loss * alpha

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")
        x = x.view((len(x), 3, 31, 31))
        return self.model(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        #raise NotImplementedError("You need to write this part!")
        y_hat = self.forward(x)
        #print(y_hat.shape)
        #y = F.one_hot(y, 4).float()
        # print(y_hat)
        # print(y)
        #print(self.loss_fn(y_hat, y))
        loss = self.loss_fn(y_hat, y) + self.l2_regular(0.001)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #raise NotImplementedError("You need to write this part!")
    # print(train_set)
    # print(train_labels)
    mean = train_set.mean(dim = 0)
    std = train_set.std(dim = 0)
    train_set = (train_set - mean) / std
    
    dev_set = (dev_set - mean) / std
    net = NeuralNet(0.01, nn.CrossEntropyLoss(), 2883, 4)
    count, _ = get_parameter_counts(net)
    print(count)
    #optimizer = optim.SGD(net.parameters(), net.lr)
    data_set = TensorDataset(train_set, train_labels)
    data_loader = DataLoader(data_set, batch_size, shuffle= True)
    loss_list = []
    for i in range(epochs):
        begin_loss = net.step(train_set, train_labels)
        for data_batch, label_batch in data_loader:
            net.step(data_batch, label_batch)
            #optimizer.step()
        after_loss = net.step(train_set, train_labels)
        loss_list.append([begin_loss, after_loss])
        #if (i + 1) % 5 == 0:
            #print(f"epoch:{i + 1}:[{begin_loss}, {after_loss}]")
    net.eval()
    with torch.no_grad():
        
        y_hat = net(dev_set).cpu()
        #print("yhat:", y_hat)
        predicted = torch.argmax(y_hat, dim=1)
        #print(len(predicted))
        predicted_labels = predicted.numpy().astype(int)
        #print(predicted_labels)
    return loss_list, predicted_labels, net
