# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:21:22 2019

@author: suchismitasa
"""

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

# Data Preprocessing

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trainTransform = tv.transforms.Compose(
    [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]
)
trainset = tv.datasets.MNIST(root='./data',  train=True, download=True,
                             transform=transform)
dataloader = data.DataLoader(trainset, batch_size=32, shuffle=False,
                                         num_workers=4)
testset = tv.datasets.MNIST(root='./data', train=False, download=True,
                            transform=transform)
testloader = data.DataLoader(testset, batch_size=32, shuffle=False,
                                         num_workers=2)
print('Data loaded.')


# Defining Model

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(6400, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 6400),
            nn.ReLU(True),
            nn.Unflatten(1, (16, 20, 20)),
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Defining Parameters
if __name__ == '__main__':
    num_epochs = 10
    model = Autoencoder().cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=1e-4)

    print('Start training.')
    for epoch in range(num_epochs):
        i = 0
        for data in dataloader:
            print(i)
            i += 1
            img, _ = data
            img = Variable(img).cpu()
            # ===================forward=====================
            output = model(img)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    torch.save(model.state_dict(), 'tr_model.pt')
