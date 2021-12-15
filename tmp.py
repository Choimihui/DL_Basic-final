import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn
import numpy as np

from datetime import datetime
from pytz import timezone

import ssl

from torchsummary import summary #!#
from tensorboardX import SummaryWriter #!#
ssl.create_default_https_context = ssl._create_unverified_context

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# DropBlock
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
    
    
# Linear Scheduler
class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=0.3, block_size=4), start_value=0, stop_value=0.5, nr_steps=1e3) #!#
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.dropout1 = nn.Dropout(0.2) #!#
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.dropout2 = nn.Dropout(0.3) #!#
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.dropout3 = nn.Dropout(0.4) #!#
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.dropout4 = nn.Dropout(0.5) #!#
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
        out = self.dropblock(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.dropblock(out) #!#
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.dropblock(out) #!#
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train(epoch):
    
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print("lr:", optimizer.param_groups[0]['lr']) #!#
            
            writer.add_scalar('training loss', (train_loss/(batch_idx + 1)), epoch * len(train_loader) + batch_idx) #!#
            writer.add_scalar('training accuracy', (100. * correct / total), epoch * len(train_loader) + batch_idx) #!#
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx) #!#


def test(epoch): #!#
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        
        writer.add_scalar('test loss', test_loss / (batch_idx + 1), epoch * len(test_loader) + batch_idx) #!#
        writer.add_scalar('test accuracy', 100. * correct / total, epoch * len(test_loader) + batch_idx) #!#
        
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None
    
if __name__ =='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'                # GPU Number 
    start_time = time.time()
    batch_size = 128
    learning_rate = 0.01

    now_str = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H_%M_%S')

    root_dir = './app/cifar10/'
    default_directory = './app/torch/save_models/dropblock4_0.99_2'

    writer = SummaryWriter('./runs/' + now_str + '/graph')  #!#

    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),               # Random Position Crop
        transforms.RandomHorizontalFlip(),                  # right and left flip
        transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                             std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                             std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
    ])

# automatically download
    train_dataset = datasets.CIFAR10(root=root_dir,
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=root_dir,
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,            # at Training Procedure, Data Shuffle = True
                                               num_workers=4)           # CPU loader number

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,            # at Test Procedure, Data Shuffle = False
                                              num_workers=4)            # CPU loader number
    
    
    model = ResNet(BasicBlock, [2, 2, 2, 2])

    modelses = model #!#
        
    modelses.cuda() #!#
    summary(modelses,(3,32,32)) #!# 
    
    if torch.cuda.device_count() > 0:
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    else:
        print("USE ONLY CPU!")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.8,
                                    weight_decay=1e-4,
                                    nesterov=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) #!#
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0

    checkpoint = load_checkpoint(default_directory)
    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, 180):
    
        # if epoch < 80:
        #     lr = learning_rate
        # elif epoch < 120:
        #     lr = learning_rate * 0.95
        # else:
        #     lr = lr * 0.8
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimizer.param_groups[0]['lr'] #!#
    
        train(epoch)
        save_checkpoint(default_directory, {
            'epoch': epoch,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        test(epoch)  
        
        scheduler.step() #!#

    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))