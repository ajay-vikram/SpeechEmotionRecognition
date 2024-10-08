import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.dropout4 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout5 = nn.Dropout(0.3)
        
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout6 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.bn2(self.pool2(torch.relu(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.bn3(self.pool3(torch.relu(self.conv3(x))))
        x = self.dropout3(x)
        
        x, _ = self.lstm1(x.permute(0, 2, 1))
        x = self.dropout4(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout5(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout6(x)
        
        x = torch.relu(self.fc1(x[:, -1, :]))  # Taking the last time step's output
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

if __name__ == "__main__":
    temp = torch.rand((64, 162, 1))
    model = CNN_LSTM()
    out = model(temp)
    print(out.shape)
