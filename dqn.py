import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.fc1 = nn.linear(state_dim+action_dim, 128)
        self.fc2 = nn.linear(128, 256)
        self.fc3 = nn.linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat(state, action dim=1)
        x = F.relu(self.fc1(state_aciton))
        x  = F.relu(self.fc2(x))
        return self.fc3(x)

    def forward_best_actions(self, state):
        """
            param:
                state: batch of states, shape: (N, |S|)
            return:
                best_action: a one hot representation of 
                best_q: Q(state, best_action)
        """
        best_action = 0
        best_q = float("-inf")
        for i in range(self.action_dim):
            action = []
            q = forward(states)
            if q > best_q:
                best_action = action
                best_q = q


        return best_action, best_q

   