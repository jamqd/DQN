import numpy as np
from collections import defaultdict

# mean average total reward over every trajectory collected 
def cumulative_reward(trajectories):
    return None

def cumulative_discounted_rewards(trajectories):
    """calculate the cumulative rewards for the given trajectories
    1. input: a list of trajectories is a list of tuples, one tuple being comprised of the following values, IN ORDER:
        1. current state (s)
        2. action agent chooses (a)
        3. reward (r)
        4. state agent enters after choosing a (s2)
        5. next action agent chooses from new state (a2)
    2. output: 
        1. list of cumulative reward for each list of trajectories
    """

    discount_factor = 0.9
    # print("boop", discount_factor)
    
    all_rewards = []

    for i, trajectory_list in enumerate(trajectories):
        curr_rewards = []
        
        for j, trajectory in enumerate(trajectory_list):

            state, action, reward, next_state, done = trajectory_list[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4]
            discounted_return = 0
            count = 0
            for k in range(j, len(trajectory_list)):
                discounted_return += (discount_factor ** count) * trajectory_list[k][2]
                count += 1
            
            curr_rewards.append(discounted_return)
        
        # done with the first list in all the trajectories
        all_rewards.append(curr_rewards)
    return all_rewards
