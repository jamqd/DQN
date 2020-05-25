import gym

#given environment, number of episodes and timesteps, run environment and return sarsa or sar trajectories
def collect_traj(env, episodes, timesteps=None, sarsa=True):
	sar_traj = []
	for i_episode in range(episodes):
		observation = env.reset()
		t = 0
		while timesteps == None or t < timesteps:
			env.render()
			action = env.action_space.sample()  # random sample of action space
			observation, reward, done, info = env.step(action)
			sar_traj.append([observation, action, reward])
			if done:
				print("Episode finished after {} timesteps".format(t + 1))
				break
			t = t + 1
		env.close()
	if sarsa:
		return sar_to_sarsa(sar_traj)
	else:
		return sar_traj

#convert sar to sarsa trajectories
def sar_to_sarsa(sar_traj):
	sarsa_traj = []
	for i in range(len(sar_traj)):
		if i != 0:
			sarsa_traj[len(sarsa_traj) - 1].append(sar_traj[i][0])
			sarsa_traj[len(sarsa_traj) - 1].append(sar_traj[i][1])
		if i != len(sar_traj) - 1:
			sarsa_traj.append(sar_traj[i])
	return sarsa_traj
