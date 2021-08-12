from packages import *

class CNN_2d(nn.Module):
    def __init__(self, hidden, in_channel):
        super(CNN_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels =in_channel, out_channels = 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    

class CNN_1d(nn.Module):
    def __init__(self, hidden, kernel_s):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size =kernel_s)
        self.conv2 = nn.Conv1d(32, 64, kernel_size =kernel_s)
        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
