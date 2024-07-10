
from __future__ import print_function
import argparse
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy		   as np
import os
import time

from utils_2              import progress_bar
from torchvision          import datasets, transforms
from models.quantize_util import BinarizeLinear

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=201, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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
best_acc = 0  # best test accuracy

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = args.test_batch_size, shuffle=True, **kwargs)

class MLP(nn.Module):
    def __init__(self, level):
        super(MLP, self).__init__()
        self.infl_ratio = 1
        self.level      = level
        
        self.fc1        = BinarizeLinear(28*28, 512*self.infl_ratio, self.level, ao_bit=32, w_bit=32, adc_bit=32)
        self.bn1        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh1     = nn.Hardtanh()
        
        self.fc2        = BinarizeLinear(512*self.infl_ratio, 512*self.infl_ratio, self.level, ao_bit=32, w_bit=32, adc_bit=32)

        self.bn2        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh2     = nn.Hardtanh()
        
        self.fc3        = BinarizeLinear(512*self.infl_ratio, 512*self.infl_ratio, self.level, ao_bit=32, w_bit=32, adc_bit=32)


        self.bn3        = nn.BatchNorm1d(512*self.infl_ratio)
        self.htanh3     = nn.Hardtanh()

        self.fc6        = BinarizeLinear(512*self.infl_ratio, 10, self.level, ao_bit=32, w_bit=32, adc_bit=32)

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
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        x = self.fc6(x)

        return self.logsoftmax(x)


def train(epoch, model):
    model.train()
    train_loss = 0
    correct    = 0
    total      = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        end = time.time()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        train_loss   += loss.item()
        _, predicted  = outputs.max(1)
        total        += targets.size(0)
        correct      += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%' % (train_loss/(batch_idx+1), 100.*correct/total))
        
    train_acc  = 100. * correct / total
    train_loss = train_loss / batch_idx
    return train_loss, train_acc

def test(epoch, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total))
        
    # Save checkpoint.
    test_acc  = 100.*correct/total
    test_loss = test_loss/batch_idx
    acc       = 100.*correct/total
    
    state = {
        'model' : model.state_dict(),
        'acc'   : acc,
        'epoch' : epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if acc > best_acc:
        best_acc = acc
    if acc > best_acc and args.save:
        torch.save(state, os.path.join('./checkpoint/', args.save))
    return test_loss, test_acc

criterion = nn.CrossEntropyLoss()


##############################################EntryPoint##########################################
if __name__ == '__main__':
    models   = [MLP(level = 2)]
    pklFiles = [['mlp_32bitADC.pkl']]
    for model, savePklFile in zip(models, pklFiles):
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
        if args.cuda:
            torch.cuda.set_device(0)
            model.cuda()
        pklFileIter = 0
        print(args.epochs)
        for epoch in range(0, args.epochs):
            print('cur train level: ',model.level,' cur train epoch:', epoch)
            #print(model)
            #print(optimizer)

            train_loss, train_acc = train(epoch, model)
            test_loss,  test_acc  = test(epoch,  model)
            print('Epoch: %d/%d | LR: %.4f | Train Loss: %.3f | Train Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f (%.2f)' %
            (epoch+1, args.epochs, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc, best_acc))
            if (epoch + 1) % 200 == 0:
                torch.save(model.state_dict(), savePklFile[pklFileIter])
                pklFileIter += 1
        print('---------------------------- TRAINING DONE ------------------------------')
        print('--------------------------- best_acc = %.2f -----------------------------' % best_acc)
 
