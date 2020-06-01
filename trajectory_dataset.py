from torch.utils.data import Dataset
import numpy as np


# I'm assuming we're using a dataloader to sample the data and perform gradient descent on it
# so this code is unbelievably simple. 
# hopefully it's what we need.


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_replay_history):
        """
            param:
                trajectories: list of trajectories. assumes each trajectory is a list of sarsa tuples 
                max_replay_history: int indicating the max number of transitions (sarsa tuples) to store
        """
        self.transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        self.max_replay_history = max_replay_history

        self.original_trajectories = self.restructure_original(trajectories)

        if len(self.transitions) > self.max_replay_history:
            self.transitions = self.transitions[len(self.transitions) - self.max_replay_history:]
               
    def __len__(self):
        """
            param:
            return:
                number of transitions

        """
        return len(self.transitions)

    def __getitem__(self, idx):
        """
            param:
                idx: index of desired transition
            return:
                item at corresponding index in transitions
        """
        return self.transitions[idx]

    def add_data(self, trajectories):
        """
            param:
                trajectories: list of trajectories. assumes each trajectory is a list of sarsa tuples 
            return:
        """
        new_transitions = np.array([transition for trajectory in trajectories for transition in trajectory])
        if len(new_transitions) >= self.max_replay_history:
            self.transitions = new_transitions[len(new_transitions) - self.max_replay_history:]
        elif len(new_transitions) + len(self.transitions) >= self.max_replay_history:
            old_start_index = (len(self.transitions) + len(new_transitions)) - self.max_replay_history
            self.transitions = np.concatenate((self.transitions[old_start_index:],new_transitions))
        else:
            self.transitions = np.concatenate((self.transitions, new_transitions))
        self.original_trajectories = self.restructure_original(self.original_trajectories + trajectories)

    def restructure_original(self, trajectories):
        """
            param:
                trajectories: list of trajectories composed of sarsa tuples
            return:
        """
        num_transitions = sum([len(trajectory) for trajectory in trajectories])
        idx_start = num_transitions - self.max_replay_history
        if idx_start > 0:
            idx_traj = 0
            next_trajectory = trajectories[idx_traj]
            while idx_start > len(next_trajectory):
                idx_start -= len(next_trajectory)
                idx_traj += 1
                next_trajectory = trajectories[idx_traj]
            if idx_traj <= len(trajectories) - 1:
                return [trajectories[idx_traj][idx_start:]]+ trajectories[idx_traj + 1:]
            elif idx_traj > len(trajectories) - 1:
                return [] # I don't think this should ever occur
        else:
            return trajectories

    def get_trajectories(self):
        """
            param:
            return:
                trajectories in their original formatting
        """
        return self.original_trajectories