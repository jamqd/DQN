from torch.utils.data import Dataset
import numpy as np


# I'm assuming we're using a dataloader to sample the data and perform gradient descent on it
# so this code is unbelievably simple. 
# hopefully it's what we need.


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_replay_history):
        """
        trajectories: list of trajectories. assumes each trajectory is a list of sarsa tuples 
        max_replay_history: int indicating the max number of transitions (sarsa tuples) to store
        """
        self.transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        self.max_replay_history = max_replay_history

        if len(self.transitions) > self.max_replay_history:
            self.transitions = self.transitions[len(self.transitions) - self.max_replay_history:]
               
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        """
        idx: index of desired transition
        """
        return self.transitions[idx]

    def add_data(self, trajectories):
        """
        trajectories: list of trajectories. assumes each trajectory is a list of sarsa tuples 
        function creates np array and appends to current array of transitions. if number of transitions exceeds
        the max replay number, then ditch the "oldest" transitions found in the current list of transitions
        """
        new_transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        if len(new_transitions) >= self.max_replay_history:
            self.transitions = new_transitions[len(new_transitions) - self.max_replay_history:]
        elif len(new_transitions) + len(self.transitions) >= self.max_replay_history:
            old_start_index = (len(self.transitions) + len(new_transitions)) - self.max_replay_history
            self.transitions = np.concatenate((self.transitions[old_start_index:],new_transitions))
        else:
            self.transitions = np.concatenate((self.transitions, new_transitions))
