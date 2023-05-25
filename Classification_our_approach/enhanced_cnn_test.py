import os
import torch as tc
from data_loader import CovidCTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from enhanced_cnn import EnhancedCNN

def check_cuda_availability():
    if tc.cuda.is_available():
        print("CUDA is available.")
        device = tc.device("cuda")
        print("Using GPU:", tc.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")
        device = tc.device("cpu")
    return device

def k_fold_cv_dataset_split(dataset, k_folds, batch_size):
    random_state = 42
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    train_loaders = []
    val_loaders = []

    for train_idx, val_idx in kfold.split(dataset):
        train_dataset = tc.utils.data.Subset(dataset, train_idx)
        val_dataset = tc.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

train_transformer = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
])

val_transformer = transforms.Compose([
    transforms.Resize(224, antialias=True),
    transforms.CenterCrop(224)
])

def training_loop(model_name, model, optimizer, loss_function, k_folds, train_loaders, val_loaders, num_epochs):

    for fold in range(k_folds):
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        
        for epoch in range(num_epochs):

            model.train()

            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass
                inputs = batch['img']
                labels = batch['label']

                # Apply train transforms to inputs
                inputs = train_transformer(inputs)

                inputs = inputs.to(device)  # Move inputs to GPU
                labels = labels.to(device)  # Move labels to GPU

                outputs = model(inputs)
                tc.onnx.export(model, inputs, "SimpleCNN.onnx", input_names=["features"], output_names=["output"])
                # Compute loss
                loss = loss_function(outputs, labels)
                batch_loss = loss_function(outputs, labels).item()
                train_loss += batch_loss

                # Backward pass and optimization
                optimizer.zero_grad()  # zeroes out the gradients of all the model parameters.
                loss.backward()  # computes the gradients of the model's parameters with respect to the loss
                optimizer.step()  # updates the model's parameters using the computed gradients and the chosen optimization algorithm

            average_train_loss = train_loss / len(train_loader)
            
            print(f'Fold: {fold+1}/{k_folds}, Epoch: {epoch+1}/{num_epochs},'
                  f'Last Batch Train Loss: {loss.item():.4f}, Average Train Loss: {average_train_loss:.4f}')

            if os.path.isfile(f"Classification_our_approach/results/{model_name}_train_results.txt"):
                with open(f"Classification_our_approach/results/{model_name}_train_results.txt", "a") as file:
                    file.write(f'{fold+1}, {epoch+1}, '
                               f'{loss.item():.4f}, {average_train_loss:.4f}\n')
            else:
                with open(f"Classification_our_approach/results/{model_name}_train_results.txt", "a") as file:
                    file.write('Fold, Epoch, Last Batch Train Loss, Average Train Loss\n')
                    file.write(f'{fold+1}, {epoch+1}, '
                               f'{loss.item():.4f}, {average_train_loss:.4f}\n')

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
    
            f1 = f1_score(targets, predictions, average='weighted')
            auc = roc_auc_score(targets, predictions)

            print(f'Epoch: {epoch+1}/{num_epochs}, Average Val Loss: {average_val_loss:.4f}, Val Acc: {(100 * correct / total):.2f}%, F1 Score: {f1:.4f}, AUC: {auc:.4f}')

            if os.path.isfile(f"Classification_our_approach/results/{model_name}_val_results.txt"):
                with open(f"Classification_our_approach/results/{model_name}_val_results.txt", "a") as file:
                    file.write(f'{fold+1}, {epoch+1}, {average_val_loss:.4f}, {(100 * correct / total):.2f}, {f1:.4f}, {auc:.4f}\n')
            else:
                with open(f"Classification_our_approach/results/{model_name}_val_results.txt", "a") as file:
                    file.write('Fold, Epoch, Average Val Loss, Val Acc, F1 Score, AUC\n')
                    file.write(f'{fold+1}, {epoch+1}, {average_val_loss:.4f}, {(100 * correct / total):.2f}, {f1:.4f}, {auc:.4f}\n')




if __name__ == "__main__":
    device = check_cuda_availability()
    
    dataset = CovidCTDataset(root_dir=r'Data/',
                             txt_COVID=r'Classification_our_approach/CT_COVID.txt',
                             txt_NonCOVID=r'Classification_our_approach/CT_NonCOVID.txt')
    print(dataset.__len__())

    learning_rate = 0.0001
    batch_size = 10
    k_folds = 3
    num_epochs = 50

    dataset_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    model_name = f'EnhancedCNN_k{k_folds}_test_epoch{num_epochs}_batch{batch_size}'
    model = EnhancedCNN()
    model.to(device)  # Move model to GPU
    parameters = model.parameters()

    optimizer = tc.optim.Adam(parameters, lr=learning_rate)
    loss_function = tc.nn.CrossEntropyLoss().to(device)

    train_loaders, val_loaders = k_fold_cv_dataset_split(dataset, k_folds=k_folds, batch_size=batch_size)
    training_loop(model_name, model, optimizer, loss_function, k_folds, train_loaders, val_loaders, num_epochs)
