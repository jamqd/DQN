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
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
            param:
                state: batch of states, shape: (N, |S|)
                action: batch of actions, shape: (N,)
            return:
                q: Q-value, Q(state, value)
        """
        # state = torch.Tensor(state)
        # action = torch.Tensor(action)
        N = len(state)
        action_one_hot = torch.zeros(N, self.action_dim)
        action_one_hot[torch.arange(N).long(), action.long()] = 1
        state_action = torch.cat((state, action_one_hot), dim=1)
        x = F.relu(self.fc1(state_action))
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
        state = torch.Tensor(state)
        N = len(state)
        state_r = state.repeat_interleave(self.action_dim, dim=0)
        all_actions_r = self.all_actions.repeat(N)
        q = self.forward(state_r, all_actions_r)
        q = q.reshape((N, self.action_dim))
        best_action = torch.argmax(q, 1)
        best_q = torch.max(q, 1)
        return best_action, best_q

   