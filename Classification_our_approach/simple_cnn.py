import torch as tc
from data_loader import CovidCTDataset
from torch.utils.data import DataLoader
from data_split import k_fold_cv_dataset_split
from training_loop import training_loop

class SimpleCNN(tc.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer_1 = tc.nn.Sequential(
            tc.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(16),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.layer_2 = tc.nn.Sequential(
            tc.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(32),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = tc.nn.Sequential(
            tc.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(64),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_4 = tc.nn.Sequential(
            tc.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(128),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_5 = tc.nn.Sequential(
            tc.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(128),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_6 = tc.nn.Sequential(
            tc.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(256),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_7 = tc.nn.Sequential(
            tc.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            tc.nn.BatchNorm2d(256),
            tc.nn.ReLU(),
            tc.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avg_pool = tc.nn.AdaptiveAvgPool2d(1)
        self.last_layer = tc.nn.Sequential(
            tc.nn.Flatten(),
            tc.nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.avg_pool(x)
        x = x.view(-1, 256)
        x = self.last_layer(x)
        return x
    
if __name__ == '__main__':
    dataset = CovidCTDataset(root_dir='Data/',
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
