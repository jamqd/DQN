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
    parser.add_argument('--learning_rate', dest='learning_rate', required = True, help="coefficient (alpha) for controlling learning rate", type=float)
    parser.add_argument('--discount_factor', dest='discount_factor', required = True, help="cofficient (gamma) to weight importance of newer samples", type=float)
    parser.add_argument('--env_name', dest = 'env_name', required = True, help="name of the gym environment", type=str)
    parser.add_argument('--iterations', dest='iterations', required=True, help="number of training iterations", type=int)
    parser.add_argument('--episodes', dest='episodes', required=True, help = "number of episodes per iterations", type=int)
    parser.add_argument('--use_ddqn', dest='use_ddqn', required=True, help = "use ddqn instead of dqn", type=bool)
    parser.add_argument('--batch_size', dest='batch_size', required=True, help = "batch size for gradient update", type=int)
    parser.add_argument('--n_threads', dest='n_threads', required=True, help = "number of threads to use", type = int)
    args = parser.parse_args()

    train(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        env_name=args.env_name,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        use_ddqn=args.use_ddqn,
        batch_size=args.batch_size,
        n_threads=args.n_threads
    )

if __name__ == "__main__":
    main()


