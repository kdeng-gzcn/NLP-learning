import torch
import time # record

import torch.nn as nn # net

import torch.optim as optim # optim method

import torch.nn.functional as F # active funtion

# iterable tool for dataset
from torch.utils.data import DataLoader, Dataset

x = torch.tensor([1.0, 2.0, 3.0]) # 1 2 3
y = torch.zeros(5, 3)  # 00000
z = torch.ones(4, 4)   # 11111

a = torch.rand(2, 2) # each term in [0, 1]
b = torch.rand(2, 2)

c = a + b  # addition
d = a * b  # np *
e = torch.matmul(a, b)  # matrix multiplication


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x): # forward computation
        # x in R^10
        x = self.fc1(x) # layer1
        x = F.relu(x) # active func
        x = self.fc2(x) # layer2

        return x # in R^1

net = myNet()

criterion = nn.MSELoss()  # loss func
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # optim method

# data loader
inputs, targets = torch.rand(3, 10), torch.rand(3, 1) # 1d or 2d dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu' # string
# device = 'cpu'
print(device)
device = torch.device(device) # call torch

begin = time.time()
net.to(device) # put my net(W, b) on cuda
inputs, targets = inputs.to(device), targets.to(device) # put my data on cuda
end = time.time()
print("put things on cuda: ", end - begin) # record time

for epoch in range(100):
    optimizer.zero_grad()  # make gradient 0 in case cumulative
    outputs = net(inputs)  # forward computation
    loss = criterion(outputs, targets)  # loss func
    loss.backward()  # backward to compute gradient
    optimizer.step()  # update weight using gradient
    # print process:
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


