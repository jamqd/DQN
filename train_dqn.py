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
import random

def compute_loss(s, a, r, s_prime, done, dqn, discount_factor, dqn_prime=None):
    """
    param:
        s : (N, |S|)
        a : batch of of actions (N,)
        r : batch of rewards (N,)
        s_prime : (N, |S|)
        q_
    return:
        a scalar value representing the loss
    """
    N = len(s)
    q = dqn.forward(s)[torch.arange(N), a.long()]
    if dqn_prime: # using ddqn and target network
        bootstrap = dqn_prime.forward(s_prime)[torch.arange(N), dqn.forward_best_actions(s_prime)[0]]
        if done:
            target = r
        else:
            target = r + discount_factor * bootstrap
    else:
        bootstrap = dqn.forward_best_actions(s_prime)[1]
        if done:
            target = r
        else:
            target = r + discount_factor * bootstrap
    target.detach() # do not propogate gradients through target
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
    max_replay_history=1000000,
    freq_report_log=5,
    online=False,
    epsilon=0.99,
    render=False
):
    """
    param:
        learning_rate:
        
    return:
        None

    """

    print("Using learning_rate={}".format(learning_rate))
    print("Using discount_factor={}".format(discount_factor))
    print("Using env_name={}".format(env_name))
    print("Using iterations={}".format(iterations))
    print("Using episodes_per_iteration={}".format(episodes_per_iteration))
    print("Using use_ddqn={}".format(use_ddqn))
    print("Using batch_size={}".format(batch_size))
    print("Using n_threads={}".format(n_threads))
    print("Using copy_params_every={}".format(copy_params_every))
    print("Using save_model_every={}".format(save_model_every))
    print("Using max_replay_history={}".format(max_replay_history))
    print("Using freq_report_log={}".format(freq_report_log))
    print("Using epsilon={}".format(epsilon))

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
        print("DQN on GPU")
        dqn = dqn.cuda()

    if online:
        #initialize dataset
        observation = env.reset()
        replay = []
        action =  env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        terminal = 1 if done else 0
        replay.append([observation, action, reward, observation_, terminal])
        dataset = TrajectoryDataset(replay, max_replay_history=max_replay_history)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 # shuffle=True,
                                                 num_workers=n_threads,
                                                 sampler=torch.utils.data.RandomSampler(dataset),
                                                 )
        # go through episodes
        for i_episode in range(episodes_per_iteration):
            observation = env.reset()
            t = 0
            while True: #repeat
                if render:
                    env.render()
                #selecting an action
                if dqn and random.random() > epsilon:
                    action = torch.squeeze(dqn.forward_best_actions([observation])[0]).item()
                else:
                    action = env.action_space.sample()  # random sample of action space
                #carry out action, observe new reward and state
                observation_, reward, done, info = env.step(action)
                #store experience in replay memory
                terminal = 1 if done else 0
                dataset.add([observation, action, reward, observation_, terminal])
                #sample random transition from replay memory
                trans = next(iter(dataloader))
                dqn_prime = None
                if use_ddqn:
                    print("Using DDQN")
                    dqn_prime = DQN(obs_space_dim, action_space_dim)
                optimizer = optim.Adam(dqn.parameters())
                loss = compute_loss(trans[0], trans[1], trans[2], trans[3], trans[4], dqn, discount_factor, dqn_prime)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() #does the gradient update, loss computed update
                #change current state
                observation = observation_
                if done:
                    break
        env.close()
        return



    # collect trajectories with random policy
    init_trajectories = collect_trajectories(env, episodes_per_iteration, dqn=dqn)
    dataset = TrajectoryDataset(init_trajectories, max_replay_history=max_replay_history)
    # print(init_trajectories)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=n_threads,
        sampler=torch.utils.data.RandomSampler(dataset),
        )

    # randomsampler = torch.utils.data.RandomSampler(dataset,
    #     num_samples = batch_size,
    #     replacement = False
    # )

    dqn_prime=None
    if use_ddqn:
        print("Using DDQN")
        dqn_prime = DQN(obs_space_dim, action_space_dim)
        if torch.cuda.is_available():
            print("DQN Prime on GPU")
            dqn_prime = dqn_prime.cuda()

    optimizer = optim.Adam(dqn.parameters())

    # torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None)

    writer = SummaryWriter()
    for i in range(iterations):
        if torch.cuda.is_available():
            print("Iteration {}, Transitions {}, MemAlloc {}".format(i, len(dataset), torch.cuda.memory_allocated()))
        else:
            print("Iteration {}, Transitions {}".format(i, len(dataset)))
        if use_ddqn and i % copy_params_every == 0:
            dqn_prime.load_state_dict(dqn.state_dict())
        
        # fitted Q-iteration
        sarsa = next(iter(dataloader))

        N = len(sarsa)
        s = sarsa[:, :obs_space_dim]
        s = torch.reshape(s, (N, obs_space_dim))

        a = sarsa[:, obs_space_dim:obs_space_dim + 1]
        a = torch.reshape(a, (N,))
        
        r = sarsa[:, obs_space_dim + 1 : obs_space_dim + 1 + 1]
        r = torch.reshape(r, (N,))

        s_prime = sarsa[:, obs_space_dim + 1 + 1: obs_space_dim + 1 + 1 + obs_space_dim]
        s_prime = torch.reshape(s_prime, (N, obs_space_dim))

        a_prime = sarsa[:, obs_space_dim + 1 + 1 + obs_space_dim:]
        a_prime = torch.reshape(a_prime, (N,))

        print(f"sarsa {sarsa.shape} {s.shape} {a.shape} {r.shape} {s_prime.shape}")

        if torch.cuda.is_available():
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            s_prime = s_prime.cuda()

        loss = compute_loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % freq_report_log == 0:
            start_time = datetime.datetime.now()
            all_traj, avg_rewards = dataset.get_trajectories()
            q_difference = q_diff(dqn, all_traj)

            # sums = []
            # for traj in all_traj:
            #     sum_reward = sum([sarsa[2] for sarsa in traj])/len(traj)
            #     sums.append(sum_reward)
            # undiscounted_avg_reward = sum(sums)/len(sums)
            undiscounted_avg_reward = sum(avg_rewards)/len(avg_rewards)

            writer.add_scalar("QDiff", q_difference, i)
            writer.add_scalar("AvgReward", undiscounted_avg_reward, i) #calculate this reward
            
            print("Time to compute avgreward and qdiff {}".format((datetime.datetime.now() - start_time).total_seconds()))

        if i% save_model_every == 0:
            torch.save(dqn, "./models/" + str(datetime.datetime.now()).replace("-","_").replace(" ","_").replace(":",".") + ".pt")
        
        # collect trajectories
        trajectories = collect_trajectories(env, episodes_per_iteration, dqn=dqn, epsilon=np.power(epsilon, i))
        dataset.add(trajectories)
        # dataloader = torch.utils.data.DataLoader(dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=n_threads)

    env.close()

def q_diff(dqn, trajectories):
    s = [sarsa[0] for traj in trajectories for sarsa in traj]
    a = [sarsa[1] for traj in trajectories for sarsa in traj]
    N = len(s)
    if torch.cuda.is_available():
        q = dqn.forward(s).detach().cpu().numpy()[np.arange(N), a]
    else:
        q = dqn.forward(s).detach().numpy()[np.arange(N), a]
    q_empirical = qvalues.cumulative_discounted_rewards(trajectories)
    q_empirical = np.concatenate([q_t for q_t in q_empirical])
    diff = q - q_empirical
    return sum(diff) / (len(q))
