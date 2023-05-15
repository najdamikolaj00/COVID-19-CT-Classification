import torch as tc
from data_loader import CovidCTDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from simple_cnn import SimpleCNN

def k_fold_cv_dataset_split(dataset, k_folds, batch_size):
    random_state = 42
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    train_loaders = []
    val_loaders = []

    for train_idx, val_idx in kfold.split(dataset):
        train_dataset = tc.utils.data.Subset(dataset, train_idx)
        val_dataset = tc.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders

def training_loop(model, optimizer, loss_function, k_folds, train_loaders, val_loaders, num_epochs):

    for fold in range(k_folds):
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        for epoch in range(num_epochs):

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass
                inputs = batch['img']
                labels = batch['label']
                outputs = model(inputs)
                
                # Compute loss
                loss = loss_function(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad() #zeroes out the gradients of all the model parameters.
                loss.backward() #computes the gradients of the model's parameters with respect to the loss
                optimizer.step() #updates the model's parameters using the computed gradients and the chosen optimization algorithm

            print(f'Fold: {fold+1}/{k_folds}, Epoch: {epoch+1}/{num_epochs},'
            f'Train Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            #torch.no_grad ensures that no gradients are computed during this process, as we don't need them for evaluation.
        with tc.no_grad():
            for batch in val_loader:
                inputs = batch['img']
                labels = batch['label']
                outputs = model(inputs)
                
                # Compute validation loss
                val_loss += loss_function(outputs, labels).item()
                
                # Compute accuracy
                _, predicted = tc.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Val Loss: {val_loss/len(val_loader):.4f},'
            f'Val Acc: {(100 * correct / total):.2f}%')


if __name__ == "__main__":        
    dataset = CovidCTDataset(root_dir=r'Data/',
                                txt_COVID='Classification_our_approach/CT_COVID.txt',
                                txt_NonCOVID='Classification_our_approach/CT_NonCOVID.txt')
    print(dataset.__len__())

    learning_rate = 0.003
    batch_size=10
    k_folds = 5
    num_epochs = 8

    dataset_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    model = SimpleCNN()
    parameters = model.parameters()

    optimizer = tc.optim.Adam(parameters, lr=learning_rate)
    loss_function = tc.nn.CrossEntropyLoss()
    train_loaders, val_loaders = k_fold_cv_dataset_split(dataset, k_folds=k_folds, batch_size=batch_size)
    training_loop(model, optimizer, loss_function, k_folds, train_loaders, val_loaders, num_epochs)