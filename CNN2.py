import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.2) 
        
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.2)  
        
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(0.2)  
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9600, 512)  
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 8)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.dropout1(x)
        
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout2(x)
        
        x = self.pool5(self.bn5(F.relu(self.conv5(x))))
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = self.bn6(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    temp = torch.rand((64, 162, 1))
    input_shape = (temp.shape[1], temp.shape[2])
    model = CNN2()
    out = model(temp)
    print(out.shape)
