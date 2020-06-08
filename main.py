import argparse
from train_dqn import train

def main():
    """
        params:
            --learning_rate LEARNING_RATE
                        coefficient (alpha) for controlling learning rate
            --discount_factor DISCOUNT_FACTOR
                        cofficient (gamma) to weight importance of newer samples
            --env_name ENV_NAME   name of the gym environment
            --iterations ITERATIONS
                                    number of training iterations
            --episodes EPISODES   number of episodes per iterations
            --use_ddqn USE_DDQN   use ddqn instead of dqn
            --batch_size BATCH_SIZE
                                    batch size for gradient update
            --n_threads N_THREADS
                                    number of threads to use
        returns:
    """

    parser = argparse.ArgumentParser(description="train dqn and/or ddqn for comparison of approaches")
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help="coefficient (alpha) for controlling learning rate", type=float)
    parser.add_argument('--discount_factor', dest='discount_factor', default=0.99, help="cofficient (gamma) to weight importance of newer samples", type=float)
    parser.add_argument('--env_name', dest = 'env_name', default="LunarLander-v2", help="name of the gym environment", type=str)
    parser.add_argument('--iterations', dest='iterations', default=1000, help="number of training iterations", type=int)
    parser.add_argument('--episodes', dest='episodes', default=100, help = "number of episodes per iterations", type=int)
    parser.add_argument('--use_ddqn', dest='use_ddqn', default=False, help = "use ddqn instead of dqn", type=bool)
    parser.add_argument('--batch_size', dest='batch_size', default=128, help = "batch size for gradient update", type=int)
    parser.add_argument('--n_threads', dest='n_threads', default=1, help = "number of threads to use", type=int)
    parser.add_argument('--max_replay', dest='max_replay', default=1000000, help = "how many transitions to store", type=int)
    parser.add_argument('--epsilon', dest='epsilon', default=0.99, help = "epsilon to use in e greedy", type=float)
    args = parser.parse_args()

    train(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        env_name=args.env_name,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        use_ddqn=args.use_ddqn,
        batch_size=args.batch_size,
        n_threads=args.n_threads,
        max_replay_history=args.max_replay,
        epsilon=args.epsilon
    )

if __name__ == "__main__":
    main()
