import torch
from dqn import DQN
import gym
import numpy as np

def loss(s, a, r, s_prime, q_prime, dqn, dqn_prime):
    """
    param:
        s : N x |S|
        a : N x |A|
        s_prime

    return:
        a scalar value representing the loss
    """
    pass

def train(
    learning_rate=0.001,
    discount_factor=0.99,
    env_name="LunarLander-v2",
    iterations=1000
):


    env = gym.make(env_name)
    if not isinstance(env.action_space, gym.space.discrete.Discrete):
        raise ValueError

    action_space_dim = env.action_space.n
    obs_space_dim = np.prod(env.observation_space.shape)


    dqn = DQN(action_space_d)
    dqn_prime = DQN(action_space_d)


    for i in iterations:
        loss = 0
        loss.backwards()

        # collect trajectories

        # fitted Q-iteration