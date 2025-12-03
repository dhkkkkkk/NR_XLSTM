import torch
import torch.nn as nn


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,3), padding=(0,0)),
            nn.Dropout(p=0.1)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,3), padding=(0,0)),
            nn.Dropout(p=0.1)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(4,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.1)
        )
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        return x

model = CNN_Model()
x = torch.rand(1,1,50,128)
x = model(x)
print(x.shape)
