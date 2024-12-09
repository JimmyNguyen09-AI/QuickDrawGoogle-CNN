# author : Trung Thanh Nguyen(Jimmy) | 09/12/2004  | ng.trungthanh04@gmail.com
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,num_classes = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,3,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features=64*5*5,out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=num_classes),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    fake_data = torch.rand(8,1,28,28)
    model  = CNN(num_classes = 15)
    prediction = model(fake_data)
    print(prediction.shape)
