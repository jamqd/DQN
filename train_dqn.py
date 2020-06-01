import torch
from torch import optim
import torch.nn.functional as F
from dqn import DQN
import run
import gym
import numpy as np
from trajectory_dataset import TrajectoryDataset
from qvalues import compute_q_value
from torch.utils.tensorboard import SummaryWriter
from run import collect_trajectories
import os
import datetime

def loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime=None):
    """
    param:
        s : (N, |S|)
        a : batch of of actions (N,)
        r : batch of rewards (N,)
        s_prime : (N, |S|)
        q_
∂
    return:
        a scalar value representing the loss
    """
    q = dqn.forward(s, a)

    if dqn_prime: # using ddqn and target network
        target = r + discount_factor * dqn_prime.forward(s_prime, dqn.forward_best_actions(s_prime)[0])
    else:
        target = r + discount_factor * dqn.forward_best_actions(s_prime)[1]
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
    n_threads=1,
    copy_params_every=100,
    save_model_every=100
):
    """
    param:
        learning_rate:
        
    return:
        None

    """

    if not os.path.isdir("./models/"):
        os.mkdir("./models/")
    env = gym.make(env_name)
    if not isinstance(env.action_space, gym.space.discrete.Discrete):
        print("Action space for env {} is not discrete".formt(env_name))
        raise ValueError

    action_space_dim = env.action_space.n
    obs_space_dim = np.prod(env.observation_space.shape)


    dqn = DQN(action_space_dim)

    init_trajectories = collect_trajectories(env, episodes_per_iteration, ddqn=use_ddqn) # fill later
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
        if use_ddqn and i % copy_params_every == 0:
            dqn_prime.load_state_dict(dqn.state_dict())

        # collect trajectories
        trajectories = collect_trajectories(env, episodes_per_iteration, ddqn=use_ddqn)
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

        all_traj = TrajectoryDataset(get_trajectories)
        q_difference = q_diff(dqn, all_traj)
        undiscounted_avg_reward = sum([(sum(sarsa[2] for sarsa in traj)/len(traj)) for traj in all_traj])/len(all_trajectories)

        writer.add_scalar("QDiff", q_difference)
        writer.add_scalar("AvgReward", undiscounted_avg_reward) #calculate this reward
        
        if save_model_every % i == 0:
            torch.save(dqn, "./models/" + str(datetime.datetime.now()).replace("-","_").replace(" ","_").replace(":",".") + ".pt")


    env.close()

def q_diff(dqn, trajectories):
    s = [sarsa[0] for sarsa in traj for traj in trajectories]
    a = [sarsa[1] for sarsa in traj for traj in trajectories]
    q = dqn.forward(s, a)
    q_empirical = compute_q_value(trajectories)
    diff = abs(q - q_empirical)
    return sum(diff) / (len(q))



        