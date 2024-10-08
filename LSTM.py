import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 8)
        
    def forward(self, x):
        x, state = self.lstm1(x)
        x, state = self.lstm2(x)
        x = self.fc(x[:, -1, :]) 
        return x

if __name__ == "__main__":
    temp = torch.rand((64, 2376, 1))
    model = LSTM()
    out = model(temp)
    print(out.shape)
