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
    parser.add_argument('--use_ddqn', dest='use_ddqn', action='store_true', help = "use ddqn instead of dqn")
    parser.set_defaults(use_ddqn=False)
    parser.add_argument('--batch_size', dest='batch_size', default=32, help = "batch size for gradient update", type=int)
    parser.add_argument('--n_threads', dest='n_threads', default=1, help = "number of threads to use", type=int)
    parser.add_argument('--max_replay', dest='max_replay', default=500000, help = "how many transitions to store", type=int)
    parser.add_argument('--epsilon', dest='epsilon', default=0.995, help = "epsilon to use in e greedy", type=float)
    parser.add_argument('--render', dest='render', action='store_true', help = "render environment or not")
    parser.set_defaults(render=False)
    parser.add_argument('--copy_params_every', dest='copy_params_every', default=100, help = "how often to copy params of neural net", type=int)
    parser.add_argument('--save_model_every', dest='save_model_every', default=100, help = "how often to save model to external folder", type=int)
    parser.add_argument('--freq_report_log', dest='freq_report_log', default=5, help = "how often to log information", type=int)
    parser.add_argument('--offline', dest='online', action='store_false', help = "online vs offline training")
    parser.set_defaults(online=True)
    parser.add_argument('--eval_episodes', dest='eval_episodes', default=16, help = "number of episodes to eval", type=int)
    parser.add_argument('--gd_optimizer', dest='gd_optimizer', default="RMSprop", help = "what optimizer to use", type=str)
    parser.add_argument('--num_episodes', dest='num_episodes', default=50000, help = "number of episodes to perform", type=int)
    args = parser.argparse()

    train(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        env_name=args.env_name,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        use_ddqn=args.use_ddqn,
        batch_size=args.batch_size,
        n_threads=args.n_threads,
        copy_params_every=args.copy_params_every,
        save_model_every=args.save_model_every,
        max_replay_history=args.max_replay,
        freq_report_log=args.freq_report_log,
        online=args.online,
        epsilon=args.epsilon,
        render=args.render,
        eval_episodes=args.eval_episodes,
        gd_optimizer=args.gd_optimizer,
        num_episodes=args.num_episodes,
    )

    

if __name__ == "__main__":
    main()
