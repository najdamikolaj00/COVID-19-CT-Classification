import numpy as np
import torch as tc
from data_loader import CovidCTDataset
import torchvision.transforms as transforms
from simple_cnn import SimpleCNN

import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_auc_score

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def check_cuda_availability():
    if tc.cuda.is_available():
        print("CUDA is available.")
        device = tc.device("cuda")
        print("Using GPU:", tc.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")
        device = tc.device("cpu")
    return device

def training_loop(model, optimizer, loss_function, train_loader, val_loader, num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total_train = 0

    for epoch in range(num_epochs):
        for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            train_correct += predicted.eq(target).sum().item()
            total_train += data.size(0)


            print('Train Epoch: {} ]\tLoss: {:.6f}'.format(
                epoch, loss.item()))

    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with tc.no_grad():
        for batch_samples in val_loader:
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            inputs = data.to(device)  # Move inputs to GPU
            labels = target.to(device)  # Move labels to GPU

            outputs = model(inputs)

            # Compute validation loss
            val_loss += loss_function(outputs, labels).item()

            # Compute accuracy
            _, predicted = tc.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Val Loss: {val_loss/len(val_loader):.4f},'
            f'Val Acc: {(100 * correct / total):.2f}%')


def test(model, loss_function, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    total_test = 0

    with tc.no_grad():
        for batch_samples in test_loader:
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            loss = loss_function(output, target.long())

            test_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            test_correct += predicted.eq(target).sum().item()
            total_test += data.size(0)

    test_loss /= total_test
    test_accuracy = 100.0 * test_correct / total_test

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, test_accuracy))

if __name__ == '__main__':
    device = check_cuda_availability()    
    batchsize=10
    trainset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/valCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir=r'Data/',
                              txt_COVID=r'Classification_Authors_based_Data_Split/COVID/testCT_COVID.txt',
                              txt_NonCOVID=r'Classification_Authors_based_Data_Split/NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)
    print(trainset.__len__())
    print(valset.__len__())
    print(testset.__len__())
    model = SimpleCNN().cuda()
    optimizer = tc.optim.Adam(model.parameters(), lr=0.001)
    loss_function = tc.nn.CrossEntropyLoss()

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)


    num_epochs = 20

    training_loop(model, optimizer, loss_function, train_loader, val_loader, num_epochs)
    test(model, loss_function, test_loader)
