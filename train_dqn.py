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

def compute_loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime=None):
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
    q = torch.squeeze(dqn.forward(s, a))
    if dqn_prime: # using ddqn and target network
        bootstrap = dqn_prime.forward(s_prime, dqn.forward_best_actions(s_prime)[0])
        # if torch.cuda.is_available():
        #     target = r.cuda() + torch.squeeze(discount_factor * bootstrap)
        # else:
        target = r + torch.squeeze(discount_factor * bootstrap)
    else:
        bootstrap = dqn.forward_best_actions(s_prime)[1]
        # if torch.cuda.is_available():
        #     target = r.cuda() + torch.squeeze(discount_factor * bootstrap[0])\
        target = r + torch.squeeze(discount_factor * bootstrap[0])

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
    max_replay_history=1000000,
    freq_report_log=5,
    online=False,
    epsilon=0.99
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
        print("DQN on GPU")
        dqn = dqn.cuda()

    if online:
        #go through episodes
        for i_episode in range(episodes_per_iteration):
            observation = env.reset()
            t = 0
            replay = []
            while True: #repeat
                if render:
                    env.render()
                #selecting an action
                if dqn and random.random() > 0.1:
                    action = torch.squeeze(dqn.forward_best_actions([observation])[0]).item()
                else:
                    action = env.action_space.sample()  # random sample of action space
                #carry out action, observe new reward and state
                observation_, reward, done, info = env.step(action)
                #store experience in replay memory
                replay.append([observation, action, reward, observation_, done])
                #sample random transition from replay memory
                trans = random.choice(replay)
                #calculate target for each minibatch transition
                    target = None
                    if trans[4] is True:
                        target_ = reward
                    else:
                        target = reward + discount_factor*dqn.forward(trans[3], action)
                #train / update gradient
                dqn_prime = None
                if use_ddqn:
                    print("Using DDQN")
                    dqn_prime = DQN(obs_space_dim, action_space_dim)
                optimizer = optim.Adam(dqn.parameters())
                try:
                    loss = compute_loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime)
                except Exception as e:
                    print(e)
                    print(s, s.shape, s.squeeze(), s.squeeze().shape)
                    print(a, a.shape, a.squeeze(), a.squeeze().shape)
                    print(r, r.shape, r.squeeze(), r.squeeze().shape)
                    print(s_prime, s_prime.shape, s_prime.squeeze(), s_prime.squeeze().shape)

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
        if True:
            s = sarsa[:, :obs_space_dim]
            a = sarsa[:, obs_space_dim:obs_space_dim + 1]
            r = sarsa[:, obs_space_dim + 1 : obs_space_dim + 1 + 1]
            s_prime = sarsa[:, obs_space_dim + 1 + 1: obs_space_dim + 1 + 1 + obs_space_dim]
            a_prime = sarsa[:, obs_space_dim + 1 + 1 + obs_space_dim:]

            torch.reshape(s, (batch_size, obs_space_dim))
            torch.reshape(a, (batch_size, 1))
            torch.reshape(r, (batch_size, 1))
            torch.reshape(s_prime, (batch_size, obs_space_dim))
            torch.reshape(a_prime, (batch_size, 1))

            print(f"sarsa {sarsa.shape} {s.shape} {a.shape} {r.shape} {s_prime.shape}")

            if torch.cuda.is_available():
                s = s.cuda()
                a = a.cuda()
                r = r.cuda()
                s_prime = s_prime.cuda()


            try:
                loss = compute_loss(s, a, r, s_prime, dqn, discount_factor, dqn_prime)
            except Exception as e:
                print(e)
                print(s, s.shape, s.squeeze(), s.squeeze().shape)
                print(a, a.shape, a.squeeze(), a.squeeze().shape)
                print(r, r.shape, r.squeeze(), r.squeeze().shape)
                print(s_prime, s_prime.shape, s_prime.squeeze(), s_prime.squeeze().shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #does the gradient update, loss computed update

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
    if torch.cuda.is_available():
        q = dqn.forward(s, a).squeeze().detach().cpu().numpy()
    else:
        q = dqn.forward(s, a).squeeze().detach().numpy()
    q_empirical = qvalues.cumulative_discounted_rewards(trajectories)
    q_empirical = np.concatenate([q_t for q_t in q_empirical])
    diff = q - q_empirical
    return sum(diff) / (len(q))
