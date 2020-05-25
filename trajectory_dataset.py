from torch.utils.data import Dataset
import numpy as np


# I'm assuming we're using a dataloader to sample the data and perform gradient descent on it
# so this code is unbelievably simple. 
# hopefully it's what we need.


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_replay_history):
        self.transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        self.max_replay_history = max_replay_history

        if len(self.transitions) > self.max_replay_history:
            self.transitions = self.transitions[len(self.transitions) - self.max_replay_history:]
               
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    def add_data(self, trajectories):
        new_transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        if len(new_transitions) >= self.max_replay_history:
            self.transitions = new_transitions[len(new_transitions) - self.max_replay_history:]
        elif len(new_transitions) + len(self.transitions) >= self.max_replay_history:
            old_start_index = (len(self.transitions) + len(new_transitions)) - self.max_replay_history
            self.transitions = np.concatenate((self.transitions[old_start_index:],new_transitions))
        else:
            self.transitions = np.concatenate((self.transitions, new_transitions))
