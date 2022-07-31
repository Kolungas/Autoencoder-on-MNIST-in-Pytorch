# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:21:22 2019

@author: suchismitasa
"""

import csv
import os
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
    if os.path.exists('log.csv'):
        os.remove('log.csv')

    num_epochs = 20
    model = Autoencoder().cpu()

    # model.load_state_dict(torch.load('Models/model_e5.pt'))
    # model.eval()

    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=1e-4)

    print('Start training.')
    for epoch in range(num_epochs):
        # =====================Training================
        for data in dataloader:
            img, _ = data
            img = Variable(img).cpu()
            # ===================forward=====================
            output = model(img)
            tr_loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
        # =================Testing======================
        for data in testloader:
            img, _ = data
            img = Variable(img).cpu()
            # ===================forward=====================
            output = model(img)
            test_loss = distance(output, img)
        # ===================log========================
        with open('log.csv', 'a') as f:
            csv.writer(f).writerow([epoch, tr_loss.item(), test_loss.item()])
        print(f'epoch [{epoch+1}/{num_epochs}], loss:{round(tr_loss.item(), 4)}')
        torch.save(model.state_dict(), f'Models/model_e{epoch}.pt')
