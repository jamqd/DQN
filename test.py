import torch
from dqn import DQN

net = DQN(3,2)

out = net.forward(torch.ones(7,3), torch.zeros(7).int())
print(out)

out = net.forward(torch.ones(7,3), torch.ones(7).int())
print(out)

out = net.forward_best_actions(torch.ones(7,3))
print(out)