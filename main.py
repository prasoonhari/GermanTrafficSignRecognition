# -*- coding: utf-8 -*-
"""

## Introduction


## **Final Model**
"""



"""# Dataset Preparation

"""


# Commented out IPython magic to ensure Python compatibility.
# Load the Drive helper and mount
# https://drive.google.com/drive/folders/1gyOhJl4IUUhzWYv3be-CDk7hhni2L7YI?usp=sharing 
# If you want to you google collab - Use the above link to download the data and put it in a folder called 'assign2_dataset' in your drive.
from google.colab import drive
drive.mount('/content/drive')
# %cd  /content/drive/'My Drive'/assign2_dataset/

"""# Dataloader"""

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Commented out IPython magic to ensure Python compatibility.

from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
# %matplotlib inline


batch_size = 32
momentum = 0.9
lr = 0.001
epochs = 100
log_interval = 100


class MyDataset(Dataset):

    def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
        self.transform = transform
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.transform:
          self.X[idx] = self.transform(self.X[idx])
        return self.X[idx],self.y[idx]

transformation_one = transforms.Compose([
    transforms.RandomAffine(degrees = (-20,20),translate=(0.10,0.10)),
]) 

transformation_two = transforms.Compose([
    transforms.Grayscale(1),
]) 


train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

train_loader = torch.utils.data.DataLoader(
     torch.utils.data.ConcatDataset(
        [train_dataset,
         MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
         MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
         MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
         MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
         MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_two),]
        ), batch_size=batch_size, shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)



"""# VisualizeImage"""
print (train_dataset.__len__())
print (len(train_loader.dataset))


def show_img(img):
  plt.figure(figsize=(2,2))
  npimg=img.numpy()
  npimg = npimg
  plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.show()

# for X, y in train_dataset:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break

for n in range(2, 20):
  a,b = train_dataset.__getitem__(n)
  arr_ = np.squeeze(a) # you can give axis attribute if you wanna squeeze in specific dimension
  show_img(make_grid(a))

a,b = train_dataset.__getitem__(20)


"""# Model"""


import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB has 43 classes
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 160, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(160)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 240, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(240)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3840, 400)
        self.fc2 = nn.Linear(400, nclasses)

    def forward(self, x):



        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.conv3_drop(x)
        x = F.relu(x)


        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.conv3_drop(x)
        x = F.relu(x)


        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(-1, 3840)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

"""# Training"""

model = Net()
model.to(device)
from torchsummary import summary

# summary(model, (3,32,32))

model = Net()
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        target = target.to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        output = model(data)
        target = target.to(device)
        validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
        # scheduler.step(np.around(validation_loss,2))
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return (100. * correct) / len(val_loader.dataset)


epoch_list = []
val_acc_list= []

for epoch in range(1, epochs + 1):
    train(epoch)
    epoch_list.append(epoch)
    val_acc = validation()
    val_acc_list.append(val_acc.item())
    scheduler.step()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '.')

from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=100)

plt.title("validation plot")
plt.xlabel("Epoch")
plt.ylabel("validation")
plt.plot(epoch_list,val_acc_list)
plt.show()

"""# Evaluate and Submit to Kaggle


"""

import pickle
import pandas as pd

outfile = 'gtsrb_kaggle.csv'

output_file = open(outfile, "w")
dataframe_dict = {"Filename" : [], "ClassId": []}

test_data = torch.load('testing/test.pt')
file_ids = pickle.load(open('testing/file_ids.pkl', 'rb'))
# model = Net() # TODO: load your model here, don't forget to put it on Eval mode !
model = Net()
model.to(device)
model.load_state_dict(torch.load('model_100' + '.pth'))
model.eval()

for i, data in enumerate(test_data):
    data = data.unsqueeze(0)
    data = data.to(device)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1].item()
    file_id = file_ids[i][0:5]
    dataframe_dict['Filename'].append(file_id)
    dataframe_dict['ClassId'].append(pred)

df = pd.DataFrame(data=dataframe_dict)
df.to_csv(outfile, index=False)
print("Written to csv file {}".format(outfile))

"""# Submitting to Kaggle

Now take this csv file, download it from your Google drive and then submit it to Kaggle to check performance of your model.

### **The Main Inspiration** 

Implement the principles of CNN and utilized various architecture like alexNet to understand the functioning of a CNN architecture.

# ## **Further Models Explored**

# ## Model - 7


# We got the following  
# Found overall Validation set: Average loss: 0.0774, Accuracy: 3793/3870 (98%)

# After the initially failed data augmentation, we again tried with different transformation, particularly keeping in mind that the images that we are training have alphabets so rotating the image by an large angle might not give the desired effect.

# Used grayScale as it  Convert image to grayscale, which is  sometimes helpful for CNN model to train faster with single channel and to learn more easily the pattern of images.


# Finally we changed the following to get the best result in our ultimate Model.

# *   Increase epoch
# *   More data augmentation
# *   reduced step LR


# Changes:



# *   Type of transformation in data augmentation
# *   decay of learning rate
# """

# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 50
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# transformation_one = transforms.Compose([
#     transforms.RandomAffine(degrees = (-20,20),translate=(0.10,0.10)),
# ]) 

# transformation_two = transforms.Compose([
#     transforms.Grayscale(1),
# ])  


# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#      torch.utils.data.ConcatDataset(
#         [train_dataset,
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_two),]
#         ), batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# model = Net()
# model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
# # optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# val_loss_list= []

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         # scheduler.step(np.around(validation_loss,2))
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#     val_loss_list.append(validation_loss)
#     return (100. * correct) / len(val_loader.dataset)


# epoch_list = []
# val_acc_list= []

# for epoch in range(1, epochs + 1):
#     train(epoch)
#     epoch_list.append(epoch)
#     val_acc = validation()
#     val_acc_list.append(val_acc.item())
#     scheduler.step()
#     model_file = 'model_' + str(epoch) + '.pth'
#     torch.save(model.state_dict(), model_file)
#     print('\nSaved model to ' + model_file + '.')


# ## Model - 6

# Deciding the feature Mapping.
# In our model so far, we have used various input and output channel, but the best result was found for the values used in this model.

# Just on this point when our network contains a number of new and unusual features which improve its performance and reduce its training time,
# The size of our network might underfit which is a significant problem. So we don't have to increase the features alot all the time to get better performance if the imageset is defined.

# Referance :

# https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks


# Validation set: Average loss: 0.4136, Accuracy: 3600/3870 (93%)

# Changes : 

# *   Focus on Order of layers in model
# """

# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 50
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# transformation_one = transforms.Compose([
#     transforms.RandomRotation(20), 
# ]) 

# transformation_two = transforms.Compose([
#     transforms.RandomHorizontalFlip(1),
# ])  


# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#      torch.utils.data.ConcatDataset(
#         [train_dataset,
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_two),]
#         ), batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# nclasses = 43 # GTSRB has 43 classes
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 160, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(160)
#         self.conv2 = nn.Conv2d(160, 160, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(160)
#         self.conv3 = nn.Conv2d(160, 240, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(240)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(3840, 400)
#         self.fc2 = nn.Linear(400, nclasses)

#     def forward(self, x):



#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = self.bn1(x)
#         x = self.conv3_drop(x)
#         x = F.relu(x)


#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = self.bn2(x)
#         x = self.conv3_drop(x)
#         x = F.relu(x)


#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)


#         # x = self.conv2(x)
#         # x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         # x = self.bn2(x)
        

#         # x = self.conv3(x) 
#         # x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         # x = self.bn3(x)


#         # print (x.size())
#         x = x.view(-1, 3840)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)


# ## Model - 5


# So, for now we have removed most of the data augmentation.

# we would like to ensure that for any parameter values, the network always produces activations with the desired distribution. So for that we introduced Batch Normalisation. normalization (shifting inputs to zero-mean and unit variance) is often used as a pre-processing step to make the data comparable across features. This therefore leads to higher learning rate and better speed.

# Also, we have used max pooling to extract sharp and smooth features as we know 
# that max pooling down-sample an input representation (image, hidden-layer output matrix, etc.).

# **Observations :**

# With this model, we learned that as Dropout is meant to block information from certain neurons completely to make sure the neurons do not co-adapt. So, the batch normalization has to be after dropout otherwise you are passing information through normalization statistics.



# Referance :
# https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd

# https://arxiv.org/pdf/1502.03167.pdf

# Found Overall Validation set: Average loss: 0.4270, Accuracy: 3537/3870 (91%)

# Changes : 


# *   Changed Data Augmentation
# """

# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 50
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# transformation_one = transforms.Compose([
#     transforms.RandomRotation(20), 
# ]) 

# transformation_two = transforms.Compose([
#     transforms.RandomHorizontalFlip(1),
# ])  


# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#      torch.utils.data.ConcatDataset(
#         [train_dataset,
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_two),]
#         ), batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# nclasses = 43 # GTSRB has 43 classes
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 160, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(160)
#         self.conv2 = nn.Conv2d(160, 160, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(160)
#         self.conv3 = nn.Conv2d(160, 240, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(240)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(3840, 400)
#         self.fc2 = nn.Linear(400, nclasses)

#     def forward(self, x):



#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = self.bn1(x)
#         x = self.conv3_drop(x)
        


#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = self.bn2(x)
#         x = self.conv3_drop(x)
        


#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.bn3(x)
        

#         # print (x.size())
#         x = x.view(-1, 3840)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)


# ## Model - 4


# Tried data augmentation as it is one of the most common pre processing technique.

# We need to increased diversity in data,  but have to make sure of the The Bias-Variance tradeoff. While data augmentation does have an explicit regularization effect,
# exploiting it can actually lead to the model not learning enough resulting in poor prediction results. Thus, we can see that there is a need to try out different combinations of data augmentation 
# to find the most appropriate one for the data set of the problem statement.

# In this case having tried various data augmentation transformation, very few yielded better performance.

# Referance:

# http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf

# https://pytorch.org/vision/stable/transforms.html

# Validation set: Average loss: 0.2823, Accuracy: 3623/3870 (94%)

# changed:


# *   Data Augmentation
# """

# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 50
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# transformation_one = transforms.Compose([
#     transforms.RandomRotation(20), 
#     transforms.RandomHorizontalFlip(p=0.5),
# ]) 

# transformation_two = transforms.Compose([
#     transforms.ColorJitter(brightness=0.2,saturation=0.4) 
# ]) 
 
# transformation_four = transforms.Compose([
#      transforms.ColorJitter(contrast=0.1), 
#      transforms.RandomRotation(40),
# ]) 

# transformation_five = transforms.Compose([
#     transforms.RandomGrayscale(p=0.1)
# ])


# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#     torch.utils.data.ConcatDataset(
#         [train_dataset,
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_one),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_two),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_four),
#          MyDataset(X_path="train/X.pt", y_path="train/y.pt",transform=transformation_five),]
#         ), batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# nclasses = 43 # GTSRB has 43 classes
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 200, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(200)
#         self.conv2 = nn.Conv2d(200, 250, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(250)
#         self.conv3 = nn.Conv2d(250, 400, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(400)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(1600, 250)
#         self.fc2 = nn.Linear(250, nclasses)

#     def forward(self, x):



#         x = self.conv1(x)
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn1(x)


#         x = self.conv2(x)
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn2(x)
        

#         x = self.conv3(x) 
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn3(x)


#         # print (x.size())
#         x = x.view(-1, 1600)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)

# model = Net()
# model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
# # optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# val_loss_list= []

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         # scheduler.step(np.around(validation_loss,2))
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#     val_loss_list.append(validation_loss)
#     return (100. * correct) / len(val_loader.dataset)


# epoch_list = []
# val_acc_list= []

# for epoch in range(1, epochs + 1):
#     train(epoch)
#     epoch_list.append(epoch)
#     val_acc = validation()
#     val_acc_list.append(val_acc.item())
#     scheduler.step()
#     model_file = 'model_' + str(epoch) + '.pth'
#     torch.save(model.state_dict(), model_file)
#     print('\nSaved model to ' + model_file + '.')


# ## Model - 3

# Having tried to use Adam Algorithm as the main optimization algorithm for faster converges to test out on various transformed data and campare their loss and accuracy, But overall, it was found that adam is fast but the accuracy fluctuates a lot, where SDG has a smoother curves. Adam fails to converge to an optimal solution under the specific setting.
# But overall, it was found that adam is fast but the accuracy fluctuates a lot, where SDG has a smoother curves.
# Adam fails to converge to an optimal solution under the specific setting.

# Changes:

# *   Reverted to SGD algorithm

# Referance:

# https://medium.com/syncedreview/iclr-2019-fast-as-adam-good-as-sgd-new-optimizer-has-both-78e37e8f9a34#:~:text=SGD%20is%20a%20variant%20of,random%20selection%20of%20data%20examples.&text=Essentially%20Adam%20is%20an%20algorithm,optimization%20of%20stochastic%20objective%20functions.


# Validation set: Average loss: 0.0714, Accuracy: 3775/3870 (98%)
# """



# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 100
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# # transformation_one = transforms.Compose([
# #     transforms.RandomRotation(20), 
# # ]) 

# # transformation_two = transforms.Compose([
# #     transforms.RandomHorizontalFlip(1),
# # ]) 



# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# model = Net()
# model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
# # optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# val_loss_list= []

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         # scheduler.step(np.around(validation_loss,2))
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#     val_loss_list.append(validation_loss)
#     return (100. * correct) / len(val_loader.dataset)


# epoch_list = []
# val_acc_list= []

# for epoch in range(1, epochs + 1):
#     train(epoch)
#     epoch_list.append(epoch)
#     val_acc = validation()
#     val_acc_list.append(val_acc.item())
#     scheduler.step()
#     model_file = 'model_' + str(epoch) + '.pth'
#     torch.save(model.state_dict(), model_file)
#     print('\nSaved model to ' + model_file + '.')

 

# ## Model - 2 

# Chioce of optimal optimization algorithm.
# Using Adam Algorithm as the main optimization algorithm for faster converges to test out on various transformed data and campare their loss and accuracy, 

# Also, So far we primarily focused on optimization algorithms for how to update the weight vectors rather than on the rate at which they are being updated. Nonetheless, adjusting the learning rate is often just as important as the actual algorithm. There are a number of aspects to consider:

# Most obviously the magnitude of the learning rate matters. If it is too large, optimization diverges, if it is too small, it takes too long to train or we end up with a suboptimal result. 

# The rate of decay is just as important. If the learning rate remains large we may simply end up bouncing around the minimum and thus not reach optimality. 

# So I used the stepLR as decreasing the learning rate during training can lead to improved accuracy and might reduced overfitting of the model.


# Found the overall : Validation set: Average loss: 0.1872, Accuracy: 3755/3870 (97%)

# Changes

# *   Used Adam Algorithm as the main optimization algorithm
# *   Introduced Batch normalization.
# *   Added StepLR



# Referance:
# https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
# """



# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 100
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
#         self.transform = transform
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         if self.transform:
#           self.X[idx] = self.transform(self.X[idx])
#         return self.X[idx],self.y[idx]

# # transformation_one = transforms.Compose([
# #     transforms.RandomRotation(20), 
# # ]) 

# # transformation_two = transforms.Compose([
# #     transforms.RandomHorizontalFlip(1),
# # ]) 



# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# nclasses = 43 # GTSRB has 43 classes
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 200, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(200)
#         self.conv2 = nn.Conv2d(200, 250, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(250)
#         self.conv3 = nn.Conv2d(250, 400, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(400)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(1600, 250)
#         self.fc2 = nn.Linear(250, nclasses)

#     def forward(self, x):



#         x = self.conv1(x)
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn1(x)


#         x = self.conv2(x)
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn2(x)
        

#         x = self.conv3(x) 
#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         x = self.bn3(x)


#         # print (x.size())
#         x = x.view(-1, 1600)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)

# model = Net()
# model.to(device)

# # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# val_loss_list= []

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         # scheduler.step(np.around(validation_loss,2))
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#     val_loss_list.append(validation_loss)
#     return (100. * correct) / len(val_loader.dataset)


# epoch_list = []
# val_acc_list= []

# for epoch in range(1, epochs + 1):
#     train(epoch)
#     epoch_list.append(epoch)
#     val_acc = validation()
#     val_acc_list.append(val_acc.item())
#     scheduler.step()
#     model_file = 'model_' + str(epoch) + '.pth'
#     torch.save(model.state_dict(), model_file)
#     print('\nSaved model to ' + model_file + '.')


# ## model - initial


# Initial approch:

# As there are lots of features, adding another layer will help us extract more features, But we have to make sure that we don't add lots of layers as 
# Instead of extracting the features, it will tend to overfit the data. 

# Also we need to carefully choose the learning rate, usually for plain SGD in neural nets we can start with 0.01 and after doing some cross-validation to find an optimal value ,  we reached at 0.001



# Found the overall :
# Validation set: Average loss: 0.2978, Accuracy: 3661/3870 (95%)



# *   Added one more convolution layer
# *   Reduced Learning Rate.



# Referance:
# https://www.kaggle.com/residentmario/tuning-your-learning-rate
# """

# # Commented out IPython magic to ensure Python compatibility.

# from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from PIL import Image
# # %matplotlib inline


# batch_size = 32
# momentum = 0.9
# lr = 0.001
# epochs = 100
# log_interval = 100


# class MyDataset(Dataset):

#     def __init__(self, X_path="X.pt", y_path="y.pt"):

#         self.X = torch.load(X_path).squeeze(1)
#         self.y = torch.load(y_path).squeeze(1)
    
#     def __len__(self):
#         return self.X.size(0)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt")
# val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt")

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# nclasses = 43 # GTSRB has 43 classes

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
#         self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
#         self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(6250, 250)
#         self.fc2 = nn.Linear(250, nclasses)

#     def forward(self, x):

#         x = self.conv1(x)

#         x = F.relu(F.max_pool2d(x, 2))
#         x = F.relu(self.conv2(x)) 

#         x = F.relu(self.conv3(x)) 

#         x = F.relu(F.max_pool2d(self.conv3_drop(x), 2))
#         # print (x.size())
#         x = x.view(-1, 6250)

#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)

# model = Net()
# model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# val_loss_list= []

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data = data.to(device)
#         output = model(data)
#         target = target.to(device)
#         validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
#         # scheduler.step(np.around(validation_loss,2))
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#     val_loss_list.append(validation_loss)
#     return (100. * correct) / len(val_loader.dataset)


# epoch_list = []
# val_acc_list= []

# for epoch in range(1, epochs + 1):
#     train(epoch)
#     epoch_list.append(epoch)
#     val_acc = validation()
#     val_acc_list.append(val_acc.item())
#     model_file = 'model_' + str(epoch) + '.pth'
#     torch.save(model.state_dict(), model_file)
#     print('\nSaved model to ' + model_file + '.')


# ## Text

# Adding more layers will help you to extract more features. But we can do that upto a certain extent. There is a limit. After that, instead of extracting features, we tend to ‘overfit’ the data. Overfitting can lead to errors in some or the other form like false positives.


# I  will give you an example. Suppose we train a model for detecting cats. If all cats features are detected and we add more layers, it can start detecting the bell the cat is wearing as a part of the cat.
# """

# Initial approch:

# As there are lots of features, adding another layer will help us extract more features, But we have to make sure that we don't add lots of layers as 
# Instead of extracting the features, it will tend to overfit the data.


# https://medium.com/syncedreview/iclr-2019-fast-as-adam-good-as-sgd-new-optimizer-has-both-78e37e8f9a34#:~:text=SGD%20is%20a%20variant%20of,random%20selection%20of%20data%20examples.&text=Essentially%20Adam%20is%20an%20algorithm,optimization%20of%20stochastic%20objective%20functions.

# https://ruder.io/optimizing-gradient-descent/

# Tried to use Adam Algorithm as the main optimization algorithm for faster converges to test out on various transformed data and campare their loss and accuracy, 
# but over it was found that adam is fast but the accuracy fluctuates a lot, where SDG has a smoother curves.
# Found out the 
#  Adam fails to converge to an optimal solution under the specific setting.



# While doing data augmentation we  have to make sure that the balance between the bias and variance. While data augmentation does have an explicit regularization effect,
# exploiting it can actually lead to the model not learning enough resulting in poor prediction results. Thus, we can see that there is a need to try out different combinations of data augmentation 
# to find the most appropriate one for the data set of the problem statement.


# https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd