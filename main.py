import torch
import torch.nn as nn
from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from sklearn.model_selection import train_test_split


class Dataset(Dataset):
    # no transformations, split ratio 0.1/0.9(val/train)
    def __init__(self, mode='train', transforms=None, split_ratio=0.1):
        self.transforms = transforms
        self.mode = mode
        # load the SVHN data 
        svhn_dataset = SVHN('.', split='train', download=True)
        self.data = svhn_dataset.data.transpose((0, 2, 3, 1))
        # take the labels from the dataset
        self.targets = svhn_dataset.labels

        if mode == 'train':
            # split the training data into training and validation sets
            # test_size = split_ratio, 10% for valudation set
            train_data, val_data, train_targets, val_targets = train_test_split(
                self.data, self.targets, test_size=split_ratio, stratify=self.targets)

            if self.mode == 'train':
                self.data = train_data
                self.targets = train_targets
            elif self.mode == 'val':
                self.data = val_data
                self.targets = val_targets

    # length of the data 
    def __len__(self):
        return len(self.data)

    # function to get item given an index
    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return (sample_x, sample_y)


# set mode to 'train' for training data, convert image to tensor
train_dataset = Dataset(mode='train', transforms=ToTensor())
# set mode to 'val' for validation data, convert image to tensor
val_dataset = Dataset(mode='val', transforms=ToTensor())
# set mode to 'test' for test data, convert image to tensor
test_dataset = Dataset(mode='test', transforms=ToTensor())


# CNN neurla network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            # 3 in channels since the images are RGB
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            # leakyReLu to avoide zero gradient problem
            nn.LeakyReLU(0.2),
            # batch normalization to normalize the ouputs of the convolution
            nn.BatchNorm2d(64),
            # reducing the spital dimensions by 0.5 (32/2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 channels in since 64 out in the first layer 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # leakyReLu to avoide zero gradient problem
            nn.LeakyReLU(0.2),
            # batch normalization to normalize the ouputs of the convolution
            nn.BatchNorm2d(128),
            # reducing the spital dimensions by 0.5 (16/2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 128 channels in since 128 out in the second layer 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # leakyReLu to avoide zero gradient problem
            nn.LeakyReLU(0.2),
            # batch normalization to normalize the ouputs of the convolution
            nn.BatchNorm2d(256),
            # reducing the spital dimensions by 0.5 (8/2)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fully_connected_layers = nn.Sequential(
            # first fully connected layer 256 4x4 inputs  
            nn.Linear(256 * 4 * 4, 1024),
            # leakyReLu to avoide zero gradient problem
            nn.LeakyReLU(0.2),
            # make 50% of elements zero
            nn.Dropout(0.5),

            # second fully connected layer 1024 inputs
            nn.Linear(1024, 512),
            # leakyReLu to avoide zero gradient problem
            nn.LeakyReLU(0.2),
            # make 50% of elements zero
            nn.Dropout(0.5),

            # third fully connected layer 512 inputs into 10 classes
            nn.Linear(512, 10),
            # log probability of each class 
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)  # flattening
        x = self.fully_connected_layers(x)
        return x


model = CNN()

# decide if training the data should happen via gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# the cross entropy loss
loss_function = nn.CrossEntropyLoss()
# the ADAM optimiser function
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# batch size and number of epochs
batch_size = 64
num_epochs = 10

# loads the training dataset onto a dataloader, shuffle on to reduce bias 
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# loads the validation dataset onto a dataloader, shuffle off to allow consistent evaluation 
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# loads the test dataset onto a dataloader, shuffle off to allow consistent evaluation 
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    # start training
    model.train()
    train_loss = 0.0  # set train loss to 0
    train_correct = 0  # set the number of corectly predicted values to 0

    for images, labels in train_data_loader:
        # move the images and labels to the GPU or CPU 
        images = images.to(device)
        labels = labels.to(device)

        # forward pass through the model
        outputs = model(images)
        # compute the loss between the predicted output and real lebels
        loss = loss_function(outputs, labels)

        optimiser.zero_grad()  # reset the gradient
        loss.backward()  # backward pass
        optimiser.step()  # update the gradient after the backward pass

        # training accuracy, compare predicted labels to the labels from the data 
        _, predicted = torch.max(outputs.data, 1)
        # sum of correctly predicted 
        train_correct += (predicted == labels).sum().item()

        # calculate the training loss batch loss * number of images 
        train_loss += loss.item() * images.size(0)

    # Validation
    model.eval()
    val_loss = 0.0  # set validation loss to 0
    val_correct = 0  # set the number of corectly predicted values to 0

    with torch.no_grad():  # disable gradient computation
        for images, labels in val_data_loader:
            # move the images and labels to the GPU or CPU
            images = images.to(device)
            labels = labels.to(device)

            # forward pass through the model
            outputs = model(images)
            # compute the loss between the predicted output and real lebels
            loss = loss_function(outputs, labels)

            # training accuracy, compare predicted labels to the labels from the data 
            _, predicted = torch.max(outputs.data, 1)
            # sum of correctly predicted 
            val_correct += (predicted == labels).sum().item()

            # calculate the validation loss batch loss * number of images 
            val_loss += loss.item() * images.size(0)

    # statistics for evaluation 
    # average training loss 
    train_loss = train_loss / len(train_dataset)
    # training accuracy
    train_accuracy = train_correct / len(train_dataset)

    # average validation loss
    val_loss = val_loss / len(val_dataset)
    # validation accuracy 
    val_accuracy = val_correct / len(val_dataset)

    # print results of the stastical test for train and val data 
    print(f'Epoch {epoch + 1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# Testing
model.eval()
test_correct = 0  # set the number of corectly predicted values to 0

with torch.no_grad():  # disable gradient computation
    for images, labels in test_data_loader:
        # move the images and labels to the GPU or CPU
        images = images.to(device)
        labels = labels.to(device)

        # forward pass through the model
        outputs = model(images)

        # training accuracy, compare predicted labels to the labels from the data 
        _, predicted = torch.max(outputs.data, 1)
        # sum of correctly predicted 
        test_correct += (predicted == labels).sum().item()

# test accuracy
test_accuracy = test_correct / len(test_dataset)
# print out test accuracy 
print(f'Test Accuracy: {test_accuracy:.4f}')