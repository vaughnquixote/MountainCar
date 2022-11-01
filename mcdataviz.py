import numpy as np 
import matplotlib.pyplot as plt
import csv


def get_first_win(final_positions):
	'''
	find index of the first value corresponding to a winning value from an 
	iterable of final positions

	Params:
		final_position (list): list of final positions achieved in each 
			training epsiode
	Returns:
	    int representing the episode of the first win recorded during training
	'''
	i = 0
	for pos in final_positions:
		if float(pos) >= 0.5:
			return i 
		i += 1
	return -1

def plot_first_win_vs_alpha(alphas, fw025, fw05, fw075, fw1):
	'''
	plot the episode of the first win for varous values of epislon used while
	training the agent

	Params:
	    alphas: a list of the alpha values tested across
	    fw025: list of the first win for the various alpha values where epsilon
		    was equal to 0.25
		fw05: `` epsilon equal to 0.5
		fw075: `` epsilon equal to 0.75
		fw1: `` epsilon equal to 1
	'''

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(alphas, fw1, color="k", label="epsilon=1")
	ax.scatter(alphas, fw025, color="r", label="epsilon=0.25")
	ax.scatter(alphas, fw05, color="g", label="epsilon=0.5")
	ax.scatter(alphas, fw075, color="b", label="epsilon=0.75")

	z1 = np.polyfit(alphas, fw1, 1)
	p1 = np.poly1d(z1)
	plt.plot(alphas, p1(alphas), color="k")

	z2 = np.polyfit(alphas, fw025, 1)
	p2 = np.poly1d(z2)
	plt.plot(alphas, p2(alphas), color="r")

	z3 = np.polyfit(alphas, fw05, 1)
	p3 = np.poly1d(z3)
	plt.plot(alphas, p3(alphas), color="g")

	z4 = np.polyfit(alphas, fw075, 1)
	p4 = np.poly1d(z4)
	plt.plot(alphas, p4(alphas), color="b")

	plt.title('First Win vs Learning Rate')
	plt.xlabel('Alpha')
	plt.ylabel('First Win')
	plt.legend(loc=2)
	plt.show()

def plot_avg_reward_vs_alpha(alphas, aw025, aw05, aw075, aw1):
	'''
	plot the average reward for the given value of alpha and the various values
	of epsilon
	Params:
	    alphas: a list of the alpha values used during testing
	    aw025: list of average reward for epsilon 0.25
		aw05: `` epsilon 0.5
		aw075: `` epsilon 0.75
		aw1: `` epsilon 1
	'''

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(alphas, aw1, color="k", label="epsilon=1")
	ax.plot(alphas, aw025, color="r", label="epsilon=0.25")
	ax.plot(alphas, aw05, color="g", label="epsilon=0.5")
	ax.plot(alphas, aw075, color="b", label="epsilon=0.75")

	'''
	z1 = np.polyfit(alphas, aw1, 1)
	p1 = np.poly1d(z1)
	plt.plot(alphas, p1(alphas), color="k")

	z2 = np.polyfit(alphas, aw025, 1)
	p2 = np.poly1d(z2)
	plt.plot(alphas, p2(alphas), color="r")

	z3 = np.polyfit(alphas, aw05, 1)
	p3 = np.poly1d(z3)
	plt.plot(alphas, p3(alphas), color="g")

	z4 = np.polyfit(alphas, aw075, 1)
	p4 = np.poly1d(z4)
	plt.plot(alphas, p4(alphas), color="b")
	'''
	plt.title('Average Reward vs Learning Rate')
	plt.xlabel('Alpha')
	plt.ylabel('Average Total Reward For 100 Episodes')
	plt.legend(loc=1)
	plt.show()


def plot_reward_vs_episode(rewards):
	'''
	plot the total reward received by the agent for each
	episode
	Params:
	    rewards: a list of the rewards received by the agent for
		    each episode, chronological
	'''

	episodes = [100*(i+1) for i in range(50)]

	rewards = [float(reward) for reward in rewards]
	batched_rewards = []
	done = False
	start = 0
	stop = 100

	while not done:
		print("average for start: ", str(start), " stop: ", str(stop) + "\n" + str(sum(rewards[start:stop])/(stop-start)))
		batched_rewards.append(sum(rewards[start:stop])/(stop-start))
		start += 100
		stop += 100
		if(start == 5000):
			done = True

	print(len(episodes))
	print(len(batched_rewards))

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(episodes, batched_rewards, color="k")

	plt.title('Total Reward vs Episode')
	plt.xlabel('Episode')
	plt.ylabel('Average Total Reward For Batch of 100 Episodes')
	plt.show()

def extract_data_from_files():
	'''
	essentially the 'main' function for this script. handles opening the 
	various files and generating the plots for each file. 
	the files correspond to various values of alpha and epsilon used to train
	the RL agent.s
	'''

	fileextension = ".csv"

	filename = "trainingepisodes_"

	trained = "trained_"

	alpha = 1

	first_wins_025 = []
	first_wins_05 = []
	first_wins_075 = []
	first_wins_1 = []

	aw_025 = []
	aw_05 = []
	aw_075 = []
	aw_1 = []

	alphas = [round((i + 1)*0.1,1) for i in range(10)]

	while alpha <= 10: 
		if(alpha < 10):
			alpha_str_rep = "0" + str(alpha)
		elif(alpha == 10):
			alpha_str_rep = "10"
		# alpha, gamma, epsilon, episodes
		epsilon = 0.25
		while epsilon <= 1:

			if (abs(epsilon - 1) < 0.001):
				epsilon_str_rep = "10"
			elif (abs(epsilon - 0.75) < 0.001):
				epsilon_str_rep = "05"
			elif (abs(epsilon - 0.5) < 0.001):
				epsilon_str_rep = "025"
			elif (abs(epsilon - 0.25) < 0.001):
				epsilon_str_rep = "075"

			full_file_name = filename + alpha_str_rep + "_" + epsilon_str_rep + fileextension

			episodes = []
			rewards = []
			final_positions = []
			max_positions = []

			with open(full_file_name, mode="r") as file:
				reader = csv.reader(file, delimiter=',')
				i = 0 
				for row in reader:
					if i == 0:
						i += 1
					else:
						episodes.append(row[0])
						rewards.append(row[1])
						final_positions.append(row[2])
						max_positions.append(row[3])

			if(abs(alpha-3) < 0.0001 and abs(epsilon-0.5) < 0.0001):
				plot_reward_vs_episode(rewards)
				
			if(abs(epsilon - 1) < 0.001):
				first_wins_1.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.75) < 0.001):
				first_wins_075.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.5) < 0.001):
				first_wins_05.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.25) < 0.001):
				first_wins_025.append(get_first_win(final_positions))


			trained_episodes = []
			trained_rewards = []
			trained_final_positions = []
			trained_max_positions = []

			trained_file = trained + alpha_str_rep + "_" + epsilon_str_rep + fileextension
			with open(trained_file, mode="r") as file:
				reader = csv.reader(file, delimiter=',')
				i = 0 
				for row in reader:
					if i == 0:
						i += 1
					else:
						trained_episodes.append(int(row[0]))
						trained_rewards.append(float(row[1]))
						trained_final_positions.append(float(row[2]))
						trained_max_positions.append(float(row[3]))

			if(abs(epsilon - 1) < 0.001):
				aw_1.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.75) < 0.001):
				aw_075.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.5) < 0.001):
				aw_05.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.25) < 0.001):
				aw_025.append(sum(trained_rewards)/len(trained_rewards))


			
			epsilon += 0.25
		
		alpha += 1

	plot_first_win_vs_alpha(alphas, first_wins_025, first_wins_05, first_wins_075, first_wins_1)

	plot_avg_reward_vs_alpha(alphas, aw_025, aw_05, aw_075, aw_1)

def extract_data_from_files_SG():
	'''
	I later added another agent which trained across different values of alpha 
	and epislon than the tabular agent. This agent was trained with a 
	semi-gradient method hence the SG added to the function signature.
	
	Not the best example of modular code as this is essentially the same
	as the previous function...but had to do this ad hoc with a time crunch
	'''

	fileextension = ".csv"

	filename = "SGtrainingepisodes_"

	trained = "SGtrained_"

	alpha = 1

	first_wins_02 = []
	first_wins_04 = []
	first_wins_06 = []
	first_wins_08 = []

	aw_02 = []
	aw_04 = []
	aw_06 = []
	aw_08 = []

	alphas = [0.025, 0.05, 0.075, 0.1, 0.125]

	alphas_str = ["0025", "005", "0075", "01", "0125"]

	while alpha <= 5: 
		# if(alpha < 10):
		# 	alpha_str_rep = "0" + str(alpha)
		# elif(alpha == 10):
		# 	alpha_str_rep = "10"
		# alpha, gamma, epsilon, episodes
		alpha_str_rep = alphas_str[alpha-1]
		epsilon = 0.2
		while epsilon <= 0.8:

			if(abs(epsilon - 0.8) < 0.001):
				epsilon_str_rep = "08"
			elif(abs(epsilon - 0.6) < 0.001):
				epsilon_str_rep = "06"
			elif(abs(epsilon - 0.4) < 0.001):
				epsilon_str_rep = "04"
			elif(abs(epsilon - 0.2) < 0.001):
				epsilon_str_rep = "02"

			full_file_name = filename + alpha_str_rep + "_" + epsilon_str_rep + fileextension

			episodes = []
			rewards = []
			final_positions = []
			max_positions = []

			with open(full_file_name, mode="r") as file:
				reader = csv.reader(file, delimiter=',')
				i = 0 
				for row in reader:
					if i == 0:
						i += 1
					else:
						episodes.append(row[0])
						rewards.append(row[1])
						final_positions.append(row[2])
						max_positions.append(row[3])

			if(abs(alpha-1) < 0.0001 and abs(epsilon-0.2) < 0.0001):
				plot_reward_vs_episode(episodes, rewards)

			if(abs(epsilon - 0.8) < 0.001):
				first_wins_08.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.6) < 0.001):
				first_wins_06.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.4) < 0.001):
				first_wins_04.append(get_first_win(final_positions))

			elif(abs(epsilon - 0.2) < 0.001):
				first_wins_02.append(get_first_win(final_positions))

			trained_episodes = []
			trained_rewards = []
			trained_final_positions = []
			trained_max_positions = []

			trained_file = trained + alpha_str_rep + "_" + epsilon_str_rep + fileextension
			with open(trained_file, mode="r") as file:
				reader = csv.reader(file, delimiter=',')
				i = 0 
				for row in reader:
					if i == 0:
						i += 1
					else:
						trained_episodes.append(int(row[0]))
						trained_rewards.append(float(row[1]))
						trained_final_positions.append(float(row[2]))
						trained_max_positions.append(float(row[3]))

			if(abs(epsilon - 0.8) < 0.001):
				aw_08.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.6) < 0.001):
				aw_06.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.4) < 0.001):
				aw_04.append(sum(trained_rewards)/len(trained_rewards))

			elif(abs(epsilon - 0.2) < 0.001):
				aw_02.append(sum(trained_rewards)/len(trained_rewards))

			epsilon += 0.2
		
		alpha += 1

	plot_first_win_vs_alpha(alphas, first_wins_02, first_wins_04, first_wins_06, first_wins_08)

	plot_avg_reward_vs_alpha(alphas, aw_02, aw_04, aw_06, aw_08)


extract_data_from_files()