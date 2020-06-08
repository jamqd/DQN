from torch.utils.data import Dataset
import numpy as np
import torch
from collections.abc import Iterable

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
        # self.transitions = np.array([transition for trajectory in trajectories for transition in trajectory], dtype=float)
          
        dim = sum([len(i) if isinstance(i, Iterable) else 1 for i in trajectories[0][0]])
        self.transitions = torch.zeros([sum([len(traj) for traj in trajectories]), dim], dtype=torch.float64)
        if torch.cuda.is_available():
            self.transitions = self.transitions.cuda()

        idx = 0
        for trajectory in trajectories:
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                r = transition[2]
                s_prime = transition[3]
                a_prime = transition[4]
                
                if torch.cuda.is_available():
                    trans_tensor = torch.Tensor(np.concatenate((s,[a],[r],s_prime,[a_prime]))).cuda()
                else:
                    trans_tensor = torch.Tensor(np.concatenate((s,[a],[r],s_prime,[a_prime])))

                self.transitions[idx] = trans_tensor
                idx+=1

        self.trajectory_avg_reward = [sum([sarsa[2] for sarsa in trajectory])/len(trajectory) for trajectory in trajectories]

        self.max_replay_history = max_replay_history

        self.original_trajectories, self.trajectory_avg_reward = self.restructure_original(trajectories, self.trajectory_avg_reward)

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

    def add(self, trajectories):
        """
            param:
                trajectories: list of trajectories. assumes each trajectory is a list of sarsa tuples 
            return:
        """
        dim = sum([len(i) if isinstance(i, Iterable) else 1 for i in trajectories[0][0]])
        new_transitions = torch.zeros([sum([len(traj) for traj in trajectories]), dim], dtype=torch.float64)
        if torch.cuda.is_available():
            new_transitions = new_transitions.cuda()

        idx = 0
        for trajectory in trajectories:
            for transition in trajectory:
                s = transition[0]
                a = transition[1]
                r = transition[2]
                s_prime = transition[3]
                a_prime = transition[4]
                
                if torch.cuda.is_available():
                    trans_tensor = torch.Tensor(np.concatenate((s,[a],[r],s_prime,[a_prime]))).cuda()
                else:
                    trans_tensor = torch.Tensor(np.concatenate((s,[a],[r],s_prime,[a_prime])))

                new_transitions[idx] = trans_tensor
                idx+=1

        if len(new_transitions) >= self.max_replay_history:
            self.transitions = new_transitions[len(new_transitions) - self.max_replay_history:]
        elif len(new_transitions) + len(self.transitions) >= self.max_replay_history:
            old_start_index = (len(self.transitions) + len(new_transitions)) - self.max_replay_history
            self.transitions = torch.cat((self.transitions[old_start_index:],new_transitions))
        else:
            self.transitions = torch.cat((self.transitions, new_transitions))

        # self.trajectory_avg_reward = self.trajectory_avg_reward + [sum([sarsa[2] for sarsa in trajectory])/len(trajectory) for trajectory in trajectories]
        self.trajectory_avg_reward = self.trajectory_avg_reward + [sum([sarsa[2] for sarsa in trajectory]) for trajectory in trajectories]
        self.original_trajectories, self.trajectory_avg_reward = self.restructure_original(self.original_trajectories + trajectories, self.trajectory_avg_reward)

    def restructure_original(self, trajectories, traj_avg_reward):
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
                clipped_portion = trajectories[idx_traj][idx_start:]
                if clipped_portion:
                    return [clipped_portion] + trajectories[idx_traj + 1:], traj_avg_reward[idx_traj:]
                    # return [clipped_portion] + trajectories[idx_traj + 1:], [sum([sarsa[2] for sarsa in clipped_portion])/len(clipped_portion)] + traj_avg_reward[idx_traj + 1:]
                else:
                    return trajectories[idx_traj + 1:], traj_avg_reward[idx_traj + 1:]
            else:
                return trajectories, traj_avg_reward
        else:
            return trajectories, traj_avg_reward

    def get_trajectories(self):
        """
            param:
            return:
                trajectories in their original formatting
        """
        return self.original_trajectories, self.trajectory_avg_reward


#         Traceback (most recent call last):
#   File "./main.py", line 48, in <module>
#     main()
#   File "./main.py", line 44, in main
#     max_replay_history=args.max_replay
#   File "/home/graham/Documents/rl_project/train_dqn.py", line 87, in train
#     dataset = TrajectoryDataset(init_trajectories, max_replay_history=max_replay_history)
#   File "/home/graham/Documents/rl_project/trajectory_dataset.py", line 17, in __init__
#     self.transitions = np.array([transition for trajectory in trajectories for transition in trajectory], dtype=float)
# ValueError: setting an array element with a sequence.