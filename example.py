import gym
from gym import spaces
from gym import envs

ENV_NAME = 'Alien-v0'
NUM_TIMESTEPS = 100
NUM_EPISODES = 20

#bare minimum example of running an instance of an environment for 1000 timesteps
def example():
	env = gym.make(ENV_NAME)
	env.reset()
	for _ in range(1000):
		env.render()
		env.step(env.action_space.sample()) #take a random action
	env.close()


#what are our actions doing to the env?
#step functionn returns observation (object), reward (float), done (boolean), info (dict)
#done indicates whether episode is terminated/whether it's time to reset the env

def loop():
	env = gym.make(ENV_NAME)
	for i_episode in range(20):
		observation = env.reset()
		for t in range(100):
			env.render()
			print(observation)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	env.close()

#spaces: each environment has an action and observation space
def space():
	env = gym.make('CartPole-v0')
	print(env.action_space)
	print(env.observation_space)
	print(env.observation_space.high)
	print(env.observation_space.low)

def sample_space():
	space = spaces.Discrete(8)
	x = space.sample()
	assert space.contains(x)
	assert space.n == 8

def get_envs():
	print(envs.registry.all())

def collect_traj():
	sar_traj = []
        env = gym.make(ENV_NAME)
        for i_episode in range(NUM_EPISODES):
                observation = env.reset()
                for t in range(NUM_TIMESTEPS):
                        env.render()
                        action = env.action_space.sample() #random sample of action space
                        observation, reward, done, info = env.step(action)
                        sar_traj.append([observation, action, reward)]
			if done:
                                print("Episode finished after {} timesteps".format(t+1))
                                break
        env.close()	
	return sar_traj

#def sar_to_sarsa(sar_traj):
#	sarsa_traj = []
#	for i in range(len(sar_traj)):
#		sarsa_traj.append(sar_traj
	

def main():
	#example()
	#loop()
	#space()
	#sample_space()
	#get_envs()
	collect_traj()

if __name__ == '__main__':
	main()
