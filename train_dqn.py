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
    else:
        bootstrap = dqn.forward_best_actions(s_prime)[1]

    target = discount_factor * bootstrap
    done_mask = done  < 0.5
    target *= done_mask 
    target += r
    target = target.detach() # do not propogate graadients through targets
    return F.mse_loss(q, target.float())

def train(
    learning_rate=0.001,
    discount_factor=0.99,
    env_name="LunarLander-v2",
    iterations=50000000,
    episodes_per_iteration=100,
    use_ddqn=False,
    batch_size=32,
    n_threads=1,
    copy_params_every=100,
    save_model_every=100,
    max_replay_history=500000,
    freq_report_log=5,
    online=False,
    epsilon=0.995,
    render=False,
    eval_episodes=16,
    gd_optimizer="RMSprop",
    num_episodes=1000
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
    print("Using online={}".format(online))
    print("Using epsilon={}".format(epsilon))
    print("Using eval_episodes={}".format(eval_episodes))
    print("Using gd_optimizer={}".format(gd_optimizer))

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

    dqn_prime=None
    if use_ddqn:
        print("Using DDQN")
        dqn_prime = DQN(obs_space_dim, action_space_dim)
        if torch.cuda.is_available():
            print("DQN Prime on GPU")
            dqn_prime = dqn_prime.cuda()

    if gd_optimizer == "ADAM":
        optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    elif gd_optimizer == "SGD":
        optimizer = optim.SGD(dqn.parameters(), lr=learning_rate)
    else:
        optimizer = optim.RMSprop(dqn.parameters(), lr=learning_rate)

    summary_writer = SummaryWriter()

    # gradient step every time a transition is collected
    if online:
        # initialize dataset
        observation = env.reset()
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        terminal = 1 if done else 0
        replay = [observation, action, reward, observation_, terminal]
        dataset = TrajectoryDataset(replay, max_replay_history=max_replay_history)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 # shuffle=True,
                                                 num_workers=n_threads,
                                                 sampler=torch.utils.data.RandomSampler(dataset),
                                                 )
        # go through episodes
        for i_episode in range(num_episodes):
            observation = env.reset()
            t = 0
            while True:  # repeat
                if render:
                    env.render()
                # selecting an action
                if dqn and random.random() > epsilon:
                    action = torch.squeeze(dqn.forward_best_actions([observation])[0]).item()
                else:
                    action = env.action_space.sample()  # random sample of action space
                # carry out action, observe new reward and state
                observation_, reward, done, info = env.step(action)
                # store experience in replay memory
                terminal = 1 if done else 0
                dataset.add([observation, action, reward, observation_, terminal])
                # sample random transition from replay memory
                trans = next(iter(dataloader))
                s, a, r, s_prime, done = unpack_dataloader_sarsd(sarsd)
                if torch.cuda.is_available():
                    s = s.cuda()
                    a = a.cuda()
                    r = r.cuda()
                    s_prime = s_prime.cuda()
                    done = done.cuda()
                loss =  compute_loss(s, a, r, s_prime, done, dqn, discount_factor, dqn_prime)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # does the gradient update, loss computed update
                # change current state
                observation = observation_
                if done:
                    break

            # log evaluation metrics
            if i_episode % freq_report_log == 0:
                start_time = datetime.datetime.now()
                log_evaluate(env, dqn, eval_episodes, summary_writer)
                print("Time to compute avgreward and qdiff {}".format((datetime.datetime.now() - start_time).total_seconds()))
            if i_episode % save_model_every == 0:
                torch.save(dqn, "./models/" + str(datetime.datetime.now()).replace("-", "_").replace(" ","_").replace(":", ".") + ".pt")

        env.close()
        return

    # collect trajectories with random policy
    init_trajectories = collect_trajectories(env, episodes_per_iteration, sarsa=False, dqn=dqn)
    dataset = TrajectoryDataset(init_trajectories, max_replay_history=max_replay_history)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        num_workers=n_threads,
        sampler=torch.utils.data.RandomSampler(dataset),
        )

    for i in range(iterations):
        if torch.cuda.is_available():
            print("Iteration {}, Transitions {}, MemAlloc {}".format(i, len(dataset), torch.cuda.memory_allocated()))
        else:
            print("Iteration {}, Transitions {}".format(i, len(dataset)))
        if use_ddqn and i % copy_params_every == 0:
            print("Copying dqn to dqn_prim")
            dqn_prime.load_state_dict(dqn.state_dict())
        
        # fitted Q-iteration
        sarsd = next(iter(dataloader))
        s, a, r, s_prime, done = unpack_dataloader_sarsd(sarsd)
        print(f"sarsa {sarsa.shape} {s.shape} {a.shape} {r.shape} {s_prime.shape} {done.shape}")

        if torch.cuda.is_available():
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            s_prime = s_prime.cuda()
            done = done.cuda()

        loss = compute_loss(s, a, r, s_prime, done, dqn, discount_factor, dqn_prime)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # collect trajectories
        trajectories = collect_trajectories(env, episodes_per_iteration, sarsa=False, dqn=dqn, epsilon=np.power(epsilon, i))
        dataset.add(trajectories)


        # log evaluation metrics
        if i % freq_report_log == 0:
            start_time = datetime.datetime.now()
            log_evaluate(env, dqn, eval_episodes, summary_writer, i)
            print("Time to compute avgreward and qdiff {}".format((datetime.datetime.now() - start_time).total_seconds()))

        if i% save_model_every == 0:
            torch.save(dqn, "./models/" + str(datetime.datetime.now()).replace("-","_").replace(" ","_").replace(":",".") + ".pt")


    env.close()

def unpack_dataloader_sarsd(sarsd):
    N = len(sarsd)
    s = sarsd[:, :obs_space_dim]
    s = torch.reshape(s, (N, obs_space_dim))

    a = sarsd[:, obs_space_dim:obs_space_dim + 1]
    a = torch.reshape(a, (N,))
    
    r = sarsd[:, obs_space_dim + 1 : obs_space_dim + 1 + 1]
    r = torch.reshape(r, (N,))

    s_prime = sarsd[:, obs_space_dim + 1 + 1: obs_space_dim + 1 + 1 + obs_space_dim]
    s_prime = torch.reshape(s_prime, (N, obs_space_dim))

    done = sarsd[:, obs_space_dim + 1 + 1 + obs_space_dim: obs_space_dim + 1 + 1 + obs_space_dim + 1]
    done = torch.reshape(done, (N,))

    return s, a, r, s_prime, done


def log_evaluate(env, dqn, num_episodes, summary_writer, iteration):
    trajectories = collect_trajectories(env=env, episodes=num_episodes, dqn=dqn)

    # average reward per trajectory
    undiscounted_avg_reward = sum([sarsa[2] for traj in trajectories for sarsa in traj])/len(trajectories)
    summary_writer.add_scalar("AvgReward", undiscounted_avg_reward, iteration) 

    # absolute difference between empirical q and q from network
    q_difference = q_diff(dqn, trajectories)
    summary_writer.add_scalar("QDiff", q_difference, iteration)
   

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
