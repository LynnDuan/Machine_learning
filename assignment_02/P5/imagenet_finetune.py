import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import matplotlib.pyplot as plt
import os

NUM_EPOCH = 20 # change max_epoch to 50
learning_rate = 0.001
str_pre = 'pre'
file_name = 'imagenet_' +str(learning_rate)+'_'+str(NUM_EPOCH)+'_'+str_pre

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)
        # modfiy method
        # self.backbone = nn.Sequential(*modules)
        # self.fc1 = nn.Linear(2048, 1024)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(1024, 32)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(32, 10)


    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)
        #modfiy structure
        # out = self.fc1(out)
        # out = self.dropout(out)
        # out = self.fc2(out)
        # out = self.dropout(out)
        # return self.fc3(out)

def train():
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomHorizontalFlip(p=0.5),#add random flip
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    validset =  datasets.CIFAR10('./data',train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,#change batch size to 16
                                          shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=16,
                                          shuffle=True, num_workers=0)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().cuda()#change criterion to MSEloss
    # optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
    #                       lr=0.001, momentum=0.9)
    optimizer = optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr = learning_rate)

    
    train_losses = []
    valid_losses = []
    itr = 0
    min_loss = 100.0
    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        train_loss_set = []
        for i, data in enumerate(trainloader, 0):
            # get the train_input
            train_input, labels = data
            train_input = Variable(train_input.cuda())

            # zero the parameter gradients
            itr += 1
            model.train()#inter train
            optimizer.zero_grad()

            # forward + backward + optimize
            train_output = model(train_input)
            labels = Variable(labels.cuda())
            loss = criterion(train_output, labels)
            loss.backward()
            optimizer.step()
            train_loss_set.append(loss.item())
        
        avg_train_loss = np.mean(np.asarray(train_loss_set))
        # train_losses.append((itr, avg_train_loss))
        
            # if i % 2000 == 1999:    # print every 2000 mini-batches
                
        train_losses.append((itr,avg_train_loss))
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, loss.item()))
        #add validation while training
        model.eval()
        valid_loss_set = []

        for i, data in enumerate(validloader):
            # get the train_input
            valid_input, valid_label = data
            valid_input = Variable(valid_input.cuda())
            valid_output = model(valid_input)
            valid_label = Variable(valid_label.cuda())
            valid_loss = criterion(valid_output, valid_label)
            valid_loss_set.append(valid_loss.item())

        avg_valid_loss = np.mean(np.asarray(valid_loss_set))
        print('Valid Epoch: %d  Loss: %f' % (epoch+1, avg_valid_loss))
        valid_losses.append((itr, avg_valid_loss))
        if avg_valid_loss<min_loss:
            model_state  = model.state_dict()
            torch.save(model_state,os.path.join('.',file_name+'.pth'))
            min_loss = avg_valid_loss


    print('Finished Training')

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)
    plt.plot(train_losses[:, 0], train_losses[:, 1], label = 'training error')
    plt.plot(valid_losses[:, 0], valid_losses[:, 1], label = 'validation error')
    plt.legend()
    # plt.show()
    plt.savefig(file_name+'.jpg')




if __name__ == '__main__':
    train()
