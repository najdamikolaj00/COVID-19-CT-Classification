import numpy as np
import torch as tc
from data_loader import CovidCTDataset
import torchvision.transforms as transforms
from simple_cnn import SimpleCNN
import os
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader
from sklearn.metrics import f1_score, roc_auc_score

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

def training_loop(model, model_name, optimizer, loss_function, train_loader, val_loader, num_epochs):
    model.train()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            loss = loss_function(output, target.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Apply the learning rate scheduler
            scheduler.step()

        average_loss = epoch_loss / len(train_loader)

        print(f'Epoch: {epoch+1}/{num_epochs},' 
            f'Average Train Loss: {average_loss:.4f}')
        
        if os.path.isfile(f"Classification_Authors_Based_Data_Split/results/{model_name}_train_results.txt"):
            with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_train_results.txt", "a") as file:
                file.write(f'{epoch+1}, {average_loss:.4f}\n')
        else:
            with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_train_results.txt", "a") as file:
                file.write('Epoch, Average Train Loss\n')
                file.write(f'{epoch+1}, {average_loss:.4f}\n')


    
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        targets = []

        with tc.no_grad():
            for batch_samples in val_loader:
                data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

                inputs = data.to(device)  # Move inputs to GPU
                labels = target.to(device)  # Move labels to GPU

                outputs = model(inputs)

                # Compute validation loss
                batch_loss = loss_function(outputs, labels).item()
                val_loss += batch_loss

                # Compute accuracy
                _, predicted = tc.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predictions.extend(predicted.tolist())
                targets.extend(target.tolist())

        average_val_loss = val_loss / len(val_loader)
   
        f1 = f1_score(targets, predictions)
        auc = roc_auc_score(targets, predictions)

        print(f'Epoch: {epoch+1}/{num_epochs}, Average Val Loss: {average_val_loss:.4f}, Val Acc: {(100 * correct / total):.2f}%, F1 Score: {f1:.4f}, AUC: {auc:.4f}')

        if os.path.isfile(f"Classification_Authors_Based_Data_Split/results/{model_name}_val_results.txt"):
            with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_val_results.txt", "a") as file:
                file.write(f'{epoch+1}, {average_val_loss:.4f}, {(100 * correct / total):.2f}, {f1:.4f}, {auc:.4f}\n')
        else:
            with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_val_results.txt", "a") as file:
                file.write('Epoch, Average Val Loss, Val Acc, F1 Score, AUC\n')
                file.write(f'{epoch+1}, {average_val_loss:.4f}, {(100 * correct / total):.2f}, {f1:.4f}, {auc:.4f}\n')

def test(model, loss_function, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    total_test = 0
    predictions = []
    targets = []

    with tc.no_grad():
        for batch_samples in test_loader:
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data).to(device)
            loss = loss_function(output, target.long())

            test_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            test_correct += predicted.eq(target).sum().item()
            total_test += data.size(0)

            # Store predictions and targets for F1 score and AUC calculation
            predictions.extend(predicted.tolist())
            targets.extend(target.tolist())
    
    test_loss /= total_test
    test_accuracy = 100.0 * test_correct / total_test
    f1 = f1_score(targets, predictions)
    auc = roc_auc_score(targets, predictions)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, F1 Score: {:.4f}, AUC: {:.4f}\n'.format(
        test_loss, test_accuracy, f1, auc))

    if os.path.isfile(f"Classification_Authors_Based_Data_Split/results/{model_name}_test_results.txt"):
        with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_test_results.txt", "a") as file:
            file.write(f'{test_loss}, {test_accuracy}, {f1}, {auc}\n')
    else:
        with open(f"Classification_Authors_Based_Data_Split/results/{model_name}_test_results.txt", "a") as file:
            file.write('Average loss, Accuracy, F1 Score, AUC\n')
            file.write(f'{test_loss}, {test_accuracy}, {f1}, {auc}\n')
    
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

    num_epochs = 50
    model_name = f'SimpleCNN_epoch{num_epochs}_batch{batchsize}'
    model = SimpleCNN().to(device)

    optimizer = tc.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = tc.nn.CrossEntropyLoss()

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    training_loop(model, model_name, optimizer, loss_function, train_loader, val_loader, num_epochs)
    test(model, loss_function, test_loader)
