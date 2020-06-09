from dqn import DQN
from torch import load
from run import collect_trajectories 
import argparse
import gym


def main():
    parser = argparse.ArgumentParser(description="to visualize trained model")
    parser.add_argument("model_name", help="path to model to visualize", type=str)
    # parser.add_argument("state_dim", help="number of state dimensions", type=int)
    # parser.add_argument("obs_dim", help="number of observations dimensions", type=int)
    args = parser.parse_args()

    # dqn = DQN(args.state_dim, args.obs_dim)
    model = load(args.model_name)
    collect_trajectories(gym.make("LunarLander-v2"), dqn=model, episodes=100, render=True)


if __name__ == "__main__":
    main()

