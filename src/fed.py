import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim, hub

import torch.nn.functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--dataset')
parser.add_argument('--nworker')
parser.add_argument('--perround')
parser.add_argument('--localiter')
parser.add_argument('--up_bit')
parser.add_argument('--down_bit')
parser.add_argument('--lr')
parser.add_argument('--momentum')
parser.add_argument('--weightdecay')
parser.add_argument('--network')
args = parser.parse_args()

DEVICE = "cuda:" + args.device

DATASET = args.dataset

NWORKER = int(args.nworker)
PERROUND = int(args.perround)
LOCALITER = int(args.localiter)

params = {'batch_size': 64,
          'shuffle': True}

# Load datasets

def quantize(v, nbit):
    min_ = np.amin(v)
    max_ = np.amax(v)

    if min_ == max_:
        return v * 0.0
    else:
        nv = ((v - min_) / (max_ - min_) * (2**nbit)).astype(np.int)
        nv = nv.astype(np.float32) / (2**nbit)
        nv = nv * (max_ - min_) + min_
        return nv

UP_NBIT = int(args.up_bit)
DOWN_NBIT = int(args.down_bit)

LR = float(args.lr)
MOMENTUM = float(args.momentum)
WEIGHTDECAY = float(args.weightdecay)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

COMPENSATE = True

NETWORK = args.network

print "========================================"

print "DATASET   =", DATASET
print "DEVICE    =", DEVICE

print "NWORKER   =", NWORKER
print "PERROUND  =", PERROUND
print "LOCALITER =", LOCALITER
print "params    =", params
print "UP_NBIT   =", UP_NBIT
print "DOWN_NBIT =", DOWN_NBIT

print "LR        =", LR
print "MOMENTUM  =", MOMENTUM
print "WDECAY    =", WEIGHTDECAY

print "COMPENSATE=", COMPENSATE
print "NETWORK   =", NETWORK

print "========================================"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if DATASET == "MNIST":

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_set = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(".", train=False, download=True, transform=transform)

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
        def name(self):
            return "LeNet"

    net = LeNet().to(device)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)


if DATASET == "CIFAR10":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10(".", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(".", train=False, download=True, transform=transform)

    #if NETWORK == "ResNet":
    net = ResNet18(10).to(device)


if DATASET == "CIFAR100":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR100(".", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(".", train=False, download=True, transform=transform)

    #if NETWORK == "ResNet":
    net = ResNet18(100).to(device)


if DATASET == "SVHN":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.SVHN(".", split="train", download=True, transform=transform)
    test_set = datasets.SVHN(".", split="test", download=True, transform=transform)

    #if NETWORK == "ResNet":
    net = ResNet18(10).to(device)
    #net = hub.load('pytorch/vision', 'wide_resnet50_2', pretrained=False).to(device)



test_loader = torch.utils.data.DataLoader(test_set, **params)

# Split into multiple training set
TRAIN_SIZE = len(train_set) / NWORKER
sizes = []
sum = 0
for i in range(0, NWORKER):
	sizes.append(TRAIN_SIZE)
	sum = sum + TRAIN_SIZE
sizes[0] = sizes[0] + len(train_set)  - sum

train_sets = torch.utils.data.random_split(train_set, sizes)
data_loaders = []
for trainset in train_sets:
	data_loaders.append(torch.utils.data.DataLoader(trainset, **params))

params = list(net.parameters())

criterion = nn.CrossEntropyLoss()

local_models = {}
local_models_diff = {}

local_models[0] = []
local_models_diff[0] = []
for p in params:
    local_models[0].append(p.data.cpu().numpy())
    local_models_diff[0].append(p.data.cpu().numpy())

for i in range(1, NWORKER):
    local_models[i] = []
    local_models_diff[i] = []

    for j in range(0, len(params)):
        local_models[i].append(np.copy(local_models[0][j]))
        local_models_diff[i].append(np.copy(local_models_diff[0][j]))

global_model = []
for p in params:
    global_model.append(p.data.cpu().numpy())

optimizers = []
for i in range(0, NWORKER):
    optimizers.append(optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHTDECAY))

errors_down = {}
errors_up = {}
for i in range(0, NWORKER):
    errors_down[i] = []
    errors_up[i] = []
    for j in range(0, len(global_model)):
        errors_down[i].append(global_model[j] * 0.0)
        errors_up[i].append(global_model[j] * 0.0)

ups = 0
downs = 0

max_acc = -1

for epoch in range(200):  

    # Choose 
    choices = np.random.choice(NWORKER, PERROUND)
    for c in choices:

        print "~", c

        # initalize local model
        for i in range(0, len(global_model)):

            if DOWN_NBIT == 32:
                local_models[c][i] = np.copy(global_model[i])
                downs = downs + np.size(global_model[i])

            else:
                d = quantize(global_model[i] - local_models[c][i] + errors_down[c][i], DOWN_NBIT)  

                if COMPENSATE == True:
                    errors_down[c][i] = (global_model[i] - local_models[c][i] + errors_down[c][i]) - d
                else:
                    errors_down[c][i] = errors_down[c][i] * 0.0

                local_models[c][i] = local_models[c][i] + d
                downs = downs + 1.0 * np.size(global_model[i]) / 32 * DOWN_NBIT 
            

        for i in range(0, len(global_model)):
            #print torch.from_numpy(local_models[c][i])
            params[i].data = 1.0 * torch.from_numpy(local_models[c][i]).data.to(device)

        for iepoch in range(0, LOCALITER):
            for i, data in enumerate(data_loaders[c], 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizers[c].zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[c].step()

        for i in range(0, len(global_model)):
            if UP_NBIT == 32:
                local_models_diff[c][i] = params[i].data.cpu().numpy() - local_models[c][i]
                ups = ups + np.size(local_models_diff[c][i])
            else:
                local_models_diff[c][i] = quantize(params[i].data.cpu().numpy() - local_models[c][i] + errors_up[c][i], UP_NBIT)

                if COMPENSATE == True:
                    errors_up[c][i] = params[i].data.cpu().numpy() - local_models[c][i] + errors_up[c][i] - local_models_diff[c][i]
                else:
                    errors_up[c][i] = errors_up[c][i] * 0.0

                ups = ups + 1.0 * np.size(local_models_diff[c][i]) / 32 * UP_NBIT


    for c in choices:
        for i in range(0, len(global_model)):
            global_model[i] = global_model[i] + local_models_diff[c][i] / PERROUND


    if epoch % 1 == 0:
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            corrected = (predicted == labels).cpu().numpy()
            correct += (corrected).sum()

            #correct += sum(predicted == labels)
     
        accuracy = 1.0 * correct / total
        max_acc = max(max_acc, accuracy)
        

        print epoch, "TEST ", max_acc,  accuracy, "      ", ups * 4 / 1024 / 1024, "MB  ", downs * 4 / 1024 / 1024, "MB"


print('Finished Training')
