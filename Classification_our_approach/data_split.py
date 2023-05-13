from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

def k_fold_cv_dataset_split(dataset, batch_size=10):
    k_folds = 5
    random_state = 42 
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    train_loaders = []
    val_loaders = []

    for train_idx, val_idx in kfold.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders

if __name__ == '__main__':
    dataset = ...  # Initialize your dataset here

    train_loaders, val_loaders = k_fold_cv_dataset_split(dataset, batch_size=10)