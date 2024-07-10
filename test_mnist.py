
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import numpy as np
import pickle

from utils_2 import progress_bar, adjust_optimizer, get_loss_for_H
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.quantize_util import  BinarizeLinear_sp, BinarizeConv2d
from models.quantize_util import  BinarizeInput, HingeLoss

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class MLP(nn.Module):
    def __init__(self, level):
        super(MLP, self).__init__()
        self.infl_ratio = 1
        self.level      = level

        self.fc1        = BinarizeLinear_sp(28*28, 512*self.infl_ratio, self.level)
        self.bn1        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh1     = nn.Hardtanh()
        
        self.fc2        = BinarizeLinear_sp(512*self.infl_ratio, 512*self.infl_ratio, self.level)
        self.bn2        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh2     = nn.Hardtanh()
        
        self.fc3        = BinarizeLinear_sp(512*self.infl_ratio, 512*self.infl_ratio, self.level)
        self.bn3        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh3     = nn.Hardtanh()
        '''
        self.fc4        = BinarizeLinear_sp(512*self.infl_ratio, 512*self.infl_ratio, self.level)
        self.bn4        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh4     = nn.Hardtanh()
        
        self.fc5        = BinarizeLinear_sp(512*self.infl_ratio, 512*self.infl_ratio, self.level)
        self.bn5        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh5     = nn.Hardtanh()
        '''
        self.fc6        = BinarizeLinear_sp(512*self.infl_ratio, 10, self.level)

        self.logsoftmax = nn.LogSoftmax()
        self.drop       = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        
        x = self.fc3(x)
        #x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        '''
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.htanh4(x)
        
        x = self.fc5(x)
        x = self.drop(x)
        x = self.bn5(x)
        x = self.htanh5(x)
        '''
        x = self.fc6(x)

        return self.logsoftmax(x)

#mod_txt = repr(model)
#print(model)
#print(model['fc2.weight'].size())

#np.savetxt(f'mlpres.txt', mod_txt, delimiter=',')


def staticModel_perLevel(levelFrom, levelTo, modelName = 'mlp.pkl'):
	pklFiles = [modelName]
	iterNu   = levelFrom
	for pklFile in pklFiles:
		model   = torch.load(pklFile, map_location='cpu')
		mlp     = MLP(iterNu)
		torch.cuda.set_device(0)
		mlp.cuda()
		iterNu += 1
		#print(model)
		#print(model['fc2.weight'].size())
		#np.savetxt(f'mlpMulti.txt', mod_txt, delimiter=',')
		mlp.load_state_dict(model)
		mlp.eval()

		correct = 0
		total = 0

		with torch.no_grad():
		    for images, labels in test_loader:
		        images = images.to('cuda')
		        labels = labels.to('cuda')
       			images = Variable(images.view(-1, 28*28))
	        	outputs = mlp(images)
		        _, predicted = torch.max(outputs.data, 1)

		        predicted = predicted.to('cuda')
       			total += labels.size(0)
	        	correct += (predicted == labels).sum()

	       		accuracy = 100 * correct / total
		#print('cur Model:', pklFile, ' ||  cur level:', mlp.level)
		#print(pklFile+' TEST Accuracy: {:.6f}%'.format(accuracy
		print(float(accuracy),', ',end='')

def staticLevel_perEpoch(level=7):
	pklFiles = [f'adc_.{level}pkl',f'adc_{level}(1).pkl',f'adc_{level}(2).pkl',f'adc_{level}(3).pkl',f'adc_{level}(4).pkl']
	iterNu   = 4
	for pklFile in pklFiles:
		model = torch.load(pklFile, map_location='cpu')
		mlp = MLP(iterNu)
		if args.cuda:
			torch.cuda.set_device(0)
			mlp.cuda()
		print('cur level', mlp.level)
		iterNu += 1
		#mod_txt = repr(model)
		#print(model)
		#print(model['fc2.weight'].size())
		#np.savetxt(f'mlpres.txt', mod_txt, delimiter=',')
		mlp.load_state_dict(model)
		mlp.eval()

		correct = 0
		total = 0

		with torch.no_grad():
		    for images, labels in test_loader:
       			images = Variable(images.view(-1, 28*28))
	        	outputs = mlp(images)
		        _, predicted = torch.max(outputs.data, 1)
       			total += labels.size(0)
	        	correct += (predicted == labels).sum()

	       		accuracy = 100 * correct / total
		print(pklFile+' TEST Accuracy: {:.6f}%'.format(accuracy))

if __name__ == '__main__':
	'''	
	model_nameList = ['mlp.pkl','adc_4(4).pkl','adc_8(4).pkl','adc_9(4).pkl','adc_10(4).pkl','adc_11(4).pkl','adc_12(4).pkl']
	for i in range(1, len(model_nameList)):
		#print(f'level_{i+6}_acc=[',end='')
		staticModel_perLevel(4, 12, f'{model_nameList[i]}')
		print()
		#print(']')
	'''
	staticModel_perLevel(2,16,'./MNIST_res/mlp_1bitADC.pkl')
        
