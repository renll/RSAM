from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from numpy import *
import numpy as np 
import torch.nn.init as init

# Training settings
parser = argparse.ArgumentParser(description='Recurrent Soft Attention Model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--regularization', action='store_true', default=False,
                    help='enable regularization (Dropout)')
parser.add_argument('--initialization', action='store_true', default=False,
                    help='enable kaiming weight initialization')
parser.add_argument('--normalization', action='store_true', default=False,
                    help='enable batch normalization')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='enable verbose display')
parser.add_argument('--seq', type=int, default=6, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=16)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                          shuffle=False, num_workers=16)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

seq=args.seq
set_printoptions(threshold='nan')
torch.set_printoptions(profile="full")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nhid=256
        self.nhid0=256
        self.rnn0 = nn.LSTMCell(256,256)        
        #self.downsample=nn.Conv2d(3, 3, kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5,padding=2)
        self.conv11 = nn.Conv2d(16, 32, kernel_size=5,padding=2)
        self.convo1 = nn.Conv2d(32, 4, kernel_size=1)
        #self.conv12 = nn.Conv2d(32, 64, kernel_size=5,padding=2) 
        # self.rnn0 = nn.RNNCell(1024,1024, nonlinearity='relu')
        self.conv21 = nn.Conv2d(3,8, kernel_size=5,padding=2)
        #self.convo1 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv22 = nn.Conv2d(8, 16, kernel_size=5,padding=2)
        self.conv23 = nn.Conv2d(16, 16, kernel_size=3,padding=1)

        self.rnn1 = nn.LSTMCell(256,256) 
        # self.convo3 = nn.Conv2d(32, 16, kernel_size=1)
        self.BN1 =nn.BatchNorm2d(8)
        self.BN2 =nn.BatchNorm2d(16)

        self.BN22 =nn.BatchNorm2d(16)
        self.BN3 =nn.BatchNorm2d(32)
        self.BN02 =nn.BatchNorm2d(16)
        self.BN03 =nn.BatchNorm2d(32)
        self.BN4=nn.BatchNorm2d(64)
        self.BN5 =nn.BatchNorm2d(128)
        self.BN6 =nn.BatchNorm1d(256)
        self.BN0 =nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(2, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(192, 256)
        self.fc5 = nn.Linear(512, 256)
        # self.fc1 = nn.Linear(400, 120)
        # self.BN3 =nn.BatchNorm1d(120)
        # self.fc2 = nn.Linear(120, 84)
        # self.BN4 =nn.BatchNorm1d(84)
        # self.fc3 = nn.Linear(84, 10)         

    def forward(self, x,target):
        if args.cuda:
            hidden0=(Variable(torch.zeros(x.size(0), self.nhid0).cuda()),
                Variable(torch.zeros(x.size(0), self.nhid0).cuda()))
            hidden=(Variable(torch.zeros(x.size(0), self.nhid).cuda()),
                Variable(torch.zeros(x.size(0), self.nhid).cuda()))  
        else:
            hidden0=(Variable(torch.zeros(x.size(0), self.nhid0)),
                Variable(torch.zeros(x.size(0), self.nhid0)))
            hidden=(Variable(torch.zeros(x.size(0), self.nhid)),
                Variable(torch.zeros(x.size(0), self.nhid)))                   
        alist=[]
        output=[]
        h=[]
        c=0
        for i in range(seq):
            x1=x.clone()
            xc= F.relu(F.max_pool2d(self.BN02(self.conv1(x1)), 2))
            xc= F.relu(F.max_pool2d(self.BN03(self.conv11(xc)), 2))
            xc=self.convo1(xc)
            xr=xc.view(-1,256)
            hidden0=self.rnn0(xr,hidden0)
            lt=F.relu(self.BN0(self.fc1(hidden0[0])))
            lt=lt.view(-1,32,32)
            alist.append(lt[3])
            lt=torch.stack([lt,lt,lt],dim=1) 
            #print(xI)
            # l=Variable(lt.data)
            # l=(l-torch.mean(l,0).repeat(l.size(0),1))/torch.std(l,0).repeat(l.size(0),1)
            # #l=2*(l0-0.5)
            # l=torch.clamp(l, min=-1, max=1)
            #l=2*(l-0.5)
            #l.append(lt)
            #print(l)
            # t1=l[:,0]*2
            # t2=l[:,1]*2
            # c1=t1*7+16
            # c2=t2+16
            # c1=l[:,0]*12+16
            # c2=l[:,1]*12+16
            # alist.append((c1[8].int(),c2[8].int()))
            # c1=c1.int().cpu().data.numpy()
            # c2=c2.int().cpu().data.numpy()
            # tlist=[]
            # for j in range(x.size(0)):
            #     xt=x[j].narrow(1,c1[j]-4,8)  
            #     xt=xt.narrow(2,c2[j]-4,8)
            #     tlist.append(xt)
            # xI=torch.stack(tlist)
            # xc= F.relu(F.max_pool2d(self.BN3(self.conv21(xI)),2))
            # xc=xc.view(-1,512)
            # xc=xI.view(-1,192)
            # xr=F.relu(self.BN6(self.fc4(xc)))
            x0=Variable(x.data,requires_grad=False)
            xI=torch.mul(lt,x0)
            xc= F.relu(F.max_pool2d(self.BN1(self.conv21(xI)),2))
            xc= F.relu(F.max_pool2d(self.BN2(self.conv22(xc)),2))
            xc= F.relu(F.max_pool2d(self.BN22(self.conv23(xc)),2))
            #xc= F.relu(self.BN3(self.conv23(xc)))
            xc=xc.view(-1,256)
            # xr=F.relu(self.BN6(self.fc5(xc)))          
            # xl=F.relu(self.BN6(self.fc2(lt)))
            # xr=torch.mul(xl,xr)
            hidden=self.rnn1(xc,hidden)
            hidden0=self.rnn0(hidden[0],hidden0)
            xe= F.log_softmax(F.relu(self.fc3(hidden[0])))
            output.append(xe)
            # if self.training:
            #     _,pred = torch.max(xe.data,1)
            #     k=pred.eq(target.data).cpu().sum()
            #     k=float(k)
            #     m=k/x.size(0)
            #     c=c+m
            #     bas=c/(i+1)
            #     #print (m)
            #     if i!=0:
            #         ht=lt.register_hook(lambda grad: 1.1*(m-bas)*grad)
            #     else:
            #         ht=lt.register_hook(lambda grad: grad)
            #     h.append(ht)
            # if i!=5:
            #     lt=F.relu(self.BN0(self.fc1(hidden0[0])))
            #     l=Variable(lt.data)
            #     l=(l-torch.mean(l,0).repeat(l.size(0),1))/torch.std(l,0).repeat(l.size(0),1)
            #     #l=2*(l0-0.5)
            #     l=torch.clamp(l, min=-1, max=1)
            #l=2*(l-0.5)
        #hidden[0]=hidden0
        #print(output)
        return output,alist

   
model = Net()

if args.cuda:
    model.cuda()

def train(epoch,lr,f):
    model.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,weight_decay=0.0001)
        optimizer.zero_grad()
        output,l=model(data,target)
        loss = F.nll_loss(output[-1], target)
        for i in range(seq-1):
            loss = F.nll_loss(output[i], target)+loss
        loss.backward()
        optimizer.step()
        train_loss+=loss.data[0]
        if batch_idx % args.log_interval == 0 and args.verbose:
            print(l,file=f)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /= len(train_loader) 
    return train_loss

def test(epoch,f):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output,l= model(data,target)

        test_loss += F.nll_loss(output[-1], target).data[0]

        out = output[-1]
        for i in range(seq-1):
           out= output[i]+out
           test_loss += F.nll_loss(output[i], target).data[0]
        _,pred = torch.max(out.data,1)
        #pred = out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader)# loss function already averages over batch size
    test_accuracy=100. * correct / len(test_loader.dataset)
    print(l,file=f)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))
    return test_loss, test_accuracy

trainCurveY=[]
accCurveY=[]
CurveX=[]
lr=0.01
f=open("outputl{}.txt".format(args.seq),'w+')

for epoch in range(1, args.epochs + 1):
    train_loss=train(epoch,lr,f)
    test_loss,test_accuracy=test(epoch,f)
    CurveX.append(epoch)
    trainCurveY.append(train_loss)
    accCurveY.append(test_accuracy)
    # if epoch==20 or epoch==40:
    lr=lr*0.95
    print('############\n\n\n',file=f)
    print(CurveX,file=f)
    print(accCurveY,file=f)
    print(trainCurveY,file=f)
print(CurveX)
print(accCurveY)
print(trainCurveY)


