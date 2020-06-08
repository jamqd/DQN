import gym
import torch
import numpy as np
import random

#given environment, number of episodes and timesteps, run environment and return sarsa or sar trajectories
def collect_trajectories(env, episodes, timesteps=None, sarsa=True, dqn=None, render=False, verbose=False, epsilon=0):
	trajectories = []
	for i_episode in range(episodes):
		observation = env.reset()
		t = 0
		sar_traj = []
		while timesteps == None or t < timesteps:
			if render:
				env.render()
			if dqn and random.random() > epsilon:
				action = torch.squeeze(dqn.forward_best_actions([observation])[0]).item()
			else: 
				action = env.action_space.sample()  # random sample of action space
			observation, reward, done, info = env.step(action)
			terminal = 1 if done else 0
			sar_traj.append([observation, action, reward, terminal])
			if done:
				if verbose:
					print("Episode finished after {} timesteps".format(t + 1))
				break
			t = t + 1
		if sarsa:
			trajectories.append(sar_to_sarsa(sar_traj))
		else:
			trajectories.append([[sarsa[0],sarsa[1],sarsa[2],sarsa[3],sarsa[5]] for sarsa in sar_to_sarsa(sar_traj)])
	# return np.asarray(trajectories)
	return trajectories

#convert sar to sarsa trajectories
def sar_to_sarsa(sar_traj):
	sarsa_traj = []
	for i in range(len(sar_traj)):
		if i != 0:
			sarsa_traj[len(sarsa_traj) - 1].insert(3, sar_traj[i][0])
			sarsa_traj[len(sarsa_traj) - 1].insert(4, sar_traj[i][1])
		sarsa_traj.append(sar_traj[i])
	sarsa_traj[len(sarsa_traj) - 1].insert(3, sar_traj[len(sar_traj) - 1][0])
	sarsa_traj[len(sarsa_traj) - 1].insert(4, sar_traj[len(sar_traj) - 1][1])
	return sarsa_traj


def main():
	#example()
	#loop()
	#space()
	#sample_space()
	#get_envs()
	#collect_traj()
	# sar = [["s1", "a1", "r1", 1], ["s2", "a2", "r2", 0], ["s3", "a3", "r3", 0], ["s4", "a4", "r4", 1]]
	# sarsa = sar_to_sarsa(sar)
	# print(sarsa)
	# env = gym.make('LunarLander-v2')
	# #space()
	# traj = collect_trajectories(env, 2, render=True)
	# print(traj)
	pass




if __name__ == '__main__':
	main()