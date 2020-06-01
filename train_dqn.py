import torch
from torch import optim
import torch.nn.functional as F
from dqn import DQN
import run
import gym
import numpy as np
from trajectory_dataset import TrajectoryDataset
from torch.utils.tensorboard import SummaryWriter
from run import collect_trajectories
import os
import datetime
import qvalues

def compute_loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime=None):
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
    q = torch.squeeze(dqn.forward(s, a))
    if dqn_prime: # using ddqn and target network
        bootstrap = dqn_prime.forward(s_prime, dqn.forward_best_actions(s_prime)[0])
        target = r + torch.squeeze(discount_factor * bootstrap)
    else:
        bootstrap = dqn.forward_best_actions(s_prime)[1]
        target = r + torch.squeeze(discount_factor * bootstrap)

    target.detach() # do not propogate graidents through target
    return F.mse_loss(q, target)

def train(
    learning_rate=0.00025,
    discount_factor=0.99,
    env_name="LunarLander-v2",
    iterations=50000000,
    episodes_per_iteration=100,
    use_ddqn=False,
    batch_size=32,
    n_threads=1,
    copy_params_every=100,
    save_model_every=100,
    max_replay_history=1000000
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
    if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
        print("Action space for env {} is not discrete".formt(env_name))
        raise ValueError

    print("Using env: {}".format(env_name))

    action_space_dim = env.action_space.n
    obs_space_dim = np.prod(env.observation_space.shape)
    print("Action space dimension: {}".format(action_space_dim))
    print("Observation space dimension {}".format(obs_space_dim))

    # initializes deep Q network
    dqn = DQN(obs_space_dim, action_space_dim)
    if torch.cuda.is_available():
        dqn.to("cuda:0")

    # collect trajectories with random policy
    init_trajectories = collect_trajectories(env, episodes_per_iteration, dqn=dqn)
    dataset = TrajectoryDataset(init_trajectories, max_replay_history=max_replay_history)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads)

    dqn_prime=None
    if use_ddqn:
        print("Using DDQN")
        dqn_prime = DQN(obs_space_dim, action_space_dim)
        if torch.cuda.is_available():
            dqn_prime.to("cuda:0")

    optimizer = optim.Adam(dqn.parameters())

    writer = SummaryWriter()
    for i in range(iterations):
        print("Iteration {}".format(i))
        if use_ddqn and i % copy_params_every == 0:
            dqn_prime.load_state_dict(dqn.state_dict())
        
        # fitted Q-iteration
        for sarsa in dataloader:
            s = sarsa[:, :obs_space_dim]
            a = sarsa[:, obs_space_dim:obs_space_dim + 1]
            r = sarsa[:, obs_space_dim + 1 : obs_space_dim + 1 + 1]
            s_prime = sarsa[:, obs_space_dim + 1 + 1: obs_space_dim + 1 + 1 + obs_space_dim]
            a_prime = sarsa[:, obs_space_dim + 1 + 1 + obs_space_dim:]
            
            # print(a.shape)
            # print(r.shape)
            loss = compute_loss(s, a.squeeze(), r.squeeze(), s_prime, dqn, discount_factor, dqn_prime)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_traj = dataset.get_trajectories()
        q_difference = q_diff(dqn, all_traj)

        sums = []
        for traj in all_traj:
            sum_reward = sum([sarsa[2] for sarsa in traj])/len(traj)
            sums.append(sum_reward)
        undiscounted_avg_reward = sum(sums)/len(sums)

        writer.add_scalar("QDiff", q_difference)
        writer.add_scalar("AvgReward", undiscounted_avg_reward) #calculate this reward
        
        if i% save_model_every == 0:
            torch.save(dqn, "./models/" + str(datetime.datetime.now()).replace("-","_").replace(" ","_").replace(":",".") + ".pt")
        
        # collect trajectories
        trajectories = collect_trajectories(env, episodes_per_iteration, dqn=dqn)
        dataset.add(trajectories)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_threads)

    env.close()

def q_diff(dqn, trajectories):
    s = [sarsa[0] for traj in trajectories for sarsa in traj]
    a = [sarsa[1] for traj in trajectories for sarsa in traj]
    q = dqn.forward(s, a).squeeze().detach().numpy()
    q_empirical = qvalues.cumulative_discounted_rewards(trajectories)
    q_empirical = np.concatenate([q_t for q_t in q_empirical])
    diff = q - q_empirical
    return sum(diff) / (len(q))