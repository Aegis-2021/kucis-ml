from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
import torch
import time
import os

import aegis.manager as AM
import aegis.networks.CNN as CNN
import aegis.networks.ResNet as ResNet
import aegis.learning.study as study
import aegis.learning.test as test


#set hyper parameters
hyper_params = {
    'learning_rate':0.01,
    'batch_size':64
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('다음 기기로 학습함:', device)

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

AM.setPath('/ML/')
trainset, validset, testset = AM.splitDataset([0.75, 0.15, 0.1], random_seed = 777)

train_dl = DataLoader(trainset, batch_size = hyper_params['batch_size'], shuffle=True)
valid_dl = DataLoader(validset, batch_size = hyper_params['batch_size'], shuffle=True)
test_dl = DataLoader(testset, batch_size = hyper_params['batch_size'], shuffle=True)
#--------------------------------------------------------------------------------------------------------------

model = CNN.CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])

params_train = {
    'model':model,
    'device':device,
    'num_epochs':10,
    'optimizer':torch.optim.Adam(model.parameters(), lr=0.1),
    'loss_func':torch.nn.CrossEntropyLoss().to(device),
    'train_dl':train_dl,
    'valid_dl':valid_dl,
    'sanity_check':False,
    'lr_scheduler':ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1),
    'path2weights':'./models/weights.pt',
}

CNN_Learner = study.Learner(model, params_train)
CNN_Learner.start()
