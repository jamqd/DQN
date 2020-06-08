import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """ 
            param:
                state_dim: int representing dimension of state vector
                action_dim: int representing number of possible actions
            return:
                a DQN object
        """
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.all_actions = torch.arange(self.action_dim)
        self.fc1 = nn.Linear(state_dim , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
            self.fc3 = self.fc3.cuda()

    def forward(self, state):
        """
            param:
                state: batch of states, shape: (N, |S|)
                action: batch of actions, shape: (N,)
            return:
                q: Q-value, (N, action_dim)
        """
        if not torch.is_tensor(state):
            state = torch.Tensor(state)
        if torch.cuda.is_available():
            state = state.cuda()
        
    
        x = F.relu(self.fc1(state.float()))
        x  = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def forward_best_actions(self, state):
        """
            param:
                state: batch of states, shape: (N, |S|)
            return:
                best_action: indexes of best action, shape: (N,)
                best_q: Q(state, best_action), shpae: (N,)
        """

        q = self.forward(state)
        best_q, best_action = torch.max(q, 1)
        return best_action, best_q

   