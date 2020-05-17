import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions 
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        return F.relu(self.conv2(x))


    def forward_all_actions(self, state):
        action = []
        return self.forward(state, action)