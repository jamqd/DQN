import torch
from torch import optim
import torch.nn.functional as F
from dqn import DQN
import gym
import numpy as np
from trajectory_dataset import TrajectoryDataset
from qvalues import compute_q_value
from torch.utils.tensorboard import SummaryWriter
def loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime=None):
    """
    param:
        s : (N, |S|)
        a : batch of one hot representations of actions (N, |A|)
        r : batch of rewards (N,)
        s_prime : (N, |S|)
        q_


    return:
        a scalar value representing the loss
    """
    q = dqn.forward(s, a)

    if dqn_prime: # using ddqn
        target = 0
    else:
        target = r + discount_factor * dqn
    target.detach() # do not propogate graidents through target
    return F.mse_loss(q, target)

def train(
    learning_rate=0.001,
    discount_factor=0.99,
    env_name="LunarLander-v2",
    iterations=1000,
    episodes_per_iteration=100,
    use_ddqn=False,
    batch_size=128,
    n_threads=1
):
    """
    param:
        learning_rate:
        
    return:
        None

    """


    env = gym.make(env_name)
    if not isinstance(env.action_space, gym.space.discrete.Discrete):
        print("Action space for env {} is not discrete".formt(env_name))
        raise ValueError

    action_space_dim = env.action_space.n
    obs_space_dim = np.prod(env.observation_space.shape)


    dqn = DQN(action_space_d)

    init_trajectories = collect_trajectories() # fill later
    dataset = TrajectoryDataset(init_trajectories)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads))



    dqn_prime=None
    if use_ddqn:
        dqn_prime = DQN(action_space_d)

    optimizer = optim.Adam(dqn.parameters())

    writer = SummaryWriter()
    
    for i in iterations:
        # collect trajectories

        trajectories = collect_trajectories()
        dataset.add(trajectories)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_threads))
        
        # fitted Q-iteration
        for (s, a, r, s_prime, a_prime) in dataloader:
            loss = loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #calculate mean of the q value differences to evaluate?

        writer.add_scalar("QDiff", q_difference)
        writer.add_scalar("AvgReward", undiscounted_avg_reward)

def q_diff(dqn, trajectories):
    s = [sarsa[0] for sarsa in traj for traj in trajectories]
    a = [sarsa[1] for sarsa in traj for traj in trajectories]
    q = dqn.forward(s, a)
    q_empirical = compute_q_value(trajectories)
    diff = abs(q - q_empirical)
    return sum(diff) / (len(q))



        