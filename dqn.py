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
        
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
            self.fc3 = self.fc3.cuda()

    def forward(self, state, action, verbose=False):
        """
            param:
                state: batch of states, shape: (N, |S|)
                action: batch of actions, shape: (N,)
            return:
                q: Q-value, Q(state, value)
        """
        if not torch.is_tensor(state):
            state = torch.Tensor(state)
        if not torch.is_tensor(action):
            action = torch.Tensor(action)

        if torch.cuda.is_available():
            state = state.cuda()
            action = action.cuda()
    
        action = torch.LongTensor(action.long())

        N = len(state)
        action_one_hot = F.one_hot(action, self.action_dim)
        state_action = torch.cat((state.float(), action_one_hot.float()), dim=1)
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
        if not torch.is_tensor(state):
            state = torch.Tensor(state)
        if torch.cuda.is_available():
            state = state.cuda()

        N = len(state)
        state_r = state.repeat_interleave(self.action_dim, dim=0)
        all_actions_r = self.all_actions.repeat(N)
        q = self.forward(state_r, all_actions_r)
        q = q.reshape((N, self.action_dim))
        best_action = torch.argmax(q, 1)
        best_q = torch.max(q, 1)
        return best_action, best_q

   