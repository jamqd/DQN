from torch.utils.data import Dataset


# I'm assuming we're using a dataloader to sample the data and perform gradient descent on it
# so this code is unbelievably simple. 
# hopefully it's what we need.

MAX_REPLAY_HISTORY = 10000

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        if trajectories and len(trajectories[0]) == 5:
            self.transitions = [transition for trajectory in trajectories for transition in trajectory]
        else:
            self.transitions = trajectories

        if len(self.transitions) > MAX_REPLAY_HISTORY:
            self.transitions = self.transitions[len(self.transitions) - MAX_REPLAY_HISTORY: -1]
        
        self.replace_number = 0
    
    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    # unsure if this function is necessary but the idea of repeatedly creating a dataset hurts me inside
    # replacement policy assumes uniform sampling
    def add_transition(self, transition):
        if len(self.transitions) == MAX_REPLAY_HISTORY:
            self.transitions[self.replace_number] = transition
            self.replace_number = self.replace_number + 1 if self.replace_number < MAX_REPLAY_HISTORY else 0
        else:
            self.transitions.append(transition)
