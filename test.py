import torch
from dqn import DQN
from run import collect_traj
import gym

net = DQN(3,2)

# out = net.forward(torch.ones(7,3), torch.zeros(7).int())
# print(out)

# out = net.forward(torch.ones(7,3), torch.ones(7).int())
# print(out)

# out = net.forward_best_actions(torch.ones(7,3))
# print(out)


env = gym.make('LunarLander-v2')
collect_traj(env, 10, timesteps=None, sarsa=True, dqn=net, render=False)