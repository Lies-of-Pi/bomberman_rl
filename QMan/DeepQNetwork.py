from torch import nn
import torch
import torch.nn.functional as F
from settings import COLS, ROWS

# loading hyperparameters
with open('agent_code/QMan/hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters

# define our network architecture for the Deep-Q-Learning model
class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # each convolutional layer double the channels and reduce the playing field by two in each direction
        self.conv1 = nn.Conv2d(k_last_frames + 1, 2*(k_last_frames + 1), kernel_size=(3,3))
        self.conv2 = nn.Conv2d(2*(k_last_frames + 1), 4*(k_last_frames + 1), kernel_size=(3,3))

        # then we use three Linear layers
        self.fc1 = nn.Linear(4*(k_last_frames + 1) * (COLS - 4)*(ROWS - 4), 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias = False)
        self.fc3 = nn.Linear(84, 6, bias=False)

        # idea: we initialize everything with zeros, so all actions have the same possibility and we have to
        # learn the rewards without speculations
        #nn.init.constant_(self.fc1.weight, 0.0)
        #nn.init.constant_(self.fc2.weight, 0.0)
        #nn.init.constant_(self.fc3.weight, 0.0)
        #nn.init.constant_(self.fc1.bias, 0.0)
        #nn.init.constant_(self.fc2.bias, 0.0)
        #nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x