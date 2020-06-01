import torch
from dqn import DQN
from run import collect_trajectories
import gym

# net = DQN(3, 2)

# out = net.forward(torch.ones(7,3), torch.zeros(7).int())
# print(out)

# out = net.forward(torch.ones(7,3), torch.ones(7).int())
# print(out)

# out = net.forward_best_actions(torch.ones(7,3))
# print(out)

# net = DQN(8,4)

# env = gym.make('LunarLander-v2')
# collect_trajectories(env, 10, timesteps=None, sarsa=True, dqn=net, render=False)
# env.close()