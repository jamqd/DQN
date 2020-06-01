def dense_nn(self):
    pass

def compute_q_value(trajectories):
    """calculate the Q value for a given trajectory. 
    1. input: a trajectory is a tuple of the following values, IN ORDER:
        1. current state (s)
        2. action agent chooses (a)
        3. reward (r)
        4. state agent enters after choosing a (s2)
        5. next action agent chooses from new state (a2)
    2. output: 
        1. Q-value
    """

    self.model = None
    self.target_model = None
    
    states, actions, rewards states_next, actions_next = trajectories
    current_q = self.model.forward(states) #TODO define these / make sure they work
    next_q = self.model.forward(states_next)
    max_next = torch.max(next_q, 1)
    expected_q = rewards + gamma * max_next

    return expected_q
