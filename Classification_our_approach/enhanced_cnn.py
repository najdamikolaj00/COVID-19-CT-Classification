import torch as tc

class EnhancedCNN(tc.nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.layer1 = tc.nn.Sequential(
            tc.nn.Conv2d(3, 32, 3, 1, padding=1),
            tc.nn.BatchNorm2d(32),  # Batch normalization
            tc.nn.ReLU(True),
            tc.nn.MaxPool2d(2, 2),
            tc.nn.Dropout(0.25)  # Dropout regularization
        )
        self.layer2 = tc.nn.Sequential(
            tc.nn.Linear(401408, 128),
            tc.nn.BatchNorm1d(128),  # Batch normalization
            tc.nn.ReLU(True),
            tc.nn.Dropout(0.5)  # Dropout regularization
        )
        self.layer3 = tc.nn.Linear(128, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
