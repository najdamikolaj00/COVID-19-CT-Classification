"""
CNN architecture with modifications based on the one implemented by the authors: 
@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865}, 
  year={2020}
}
"""

import torch as tc

class SimpleCNN(tc.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer1 = tc.nn.Sequential(
            tc.nn.Conv2d(3, 32, 3, 1, padding=1),
            tc.nn.ReLU(True),
            tc.nn.MaxPool2d(2, 2)
        )
        self.layer2 = tc.nn.Sequential(
            tc.nn.Linear(32 * 16 * 16, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        return x
