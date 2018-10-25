import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
NUM_EPOCH = 10
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

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def train():
    ## Define the training dataloader
    transform = transforms.Compose([transforms.RandomCrop(50),#add random crop
                                    transforms.Resize(224),
                                    transforms.RandomHorizontalFlip(p=0.5),#add random flip
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./traindata', download=True, transform=transform)
    validset =  datasets.CIFAR10('./testdata',train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=4,
                                          shuffle=True, num_workers=2)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().cuda()#change criterion to MSEloss
    # optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
    #                       lr=0.001, momentum=0.9)
    optimizer = optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr = learning_rate)

    
    train_losses = []
    valid_losses = []
    itr = 0
    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
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
            train_losses.append((itr,loss.item()))

            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    ('Epoch: %d Itr: %d Loss: %f' %(epoch + 1, i + 1, loss.item()))
                #add validation while training
                model.eval()
                valid_loss_set = []
                valid_itr = 0
                for i, data in enumerate(trainloader, 0):
                    # get the train_input
                    valid_input, valid_label = data
                    valid_input = Variable(valid_input.cuda())
                    valid_output = model(valid_input)
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_output, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break
                    avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                    print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch, valid_itr, avg_valid_loss))
                    valid_losses.append((itr, avg_valid_loss))

    print('Finished Training')

train_losses = np.asarray(train_losses)
valid_losses = np.asarray(valid_losses)
plt.plot(train_losses[:, 0],
         train_losses[:, 1])
plt.plot(valid_losses[:, 0],
         valid_losses[:, 1])
plt.show()
plt.savefig(file_name+'.jpg')




if __name__ == '__main__':
    train()
