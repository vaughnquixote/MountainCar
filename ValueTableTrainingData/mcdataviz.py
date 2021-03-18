import numpy as np 
import random
import matplotlib.pyplot as plt
import csv


def get_first_win(final_positions):

	i = 0
	for pos in final_positions:

		if float(pos) >= 0.5:
			return i 

		i += 1

def plot_first_win_vs_alpha(alphas, fw025, fw05, fw075, fw1):

	colors = ("black","red", "green", "blue")
	data = (fw025, fw05, fw075, fw1)

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


def plot_reward_vs_episode(episodes,rewards):

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

	fileextension = ".csv"

	filename = "trainingepisodes_"

	trained = "trained_"

	alpha = 1

	trainingepisodes = dict()

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

			if(abs(epsilon - 1) < 0.001):
				epsilon_str_rep = "10"
			elif(abs(epsilon - 0.75) < 0.001):
				epsilon_str_rep = "05"
			elif(abs(epsilon - 0.5) < 0.001):
				epsilon_str_rep = "025"
			elif(abs(epsilon - 0.25) < 0.001):
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
			 	plot_reward_vs_episode(episodes, rewards)



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

	# plot_first_win_vs_alpha(alphas, first_wins_025, first_wins_05, first_wins_075, first_wins_1)

	# plot_avg_reward_vs_alpha(alphas, aw_025, aw_05, aw_075, aw_1)


extract_data_from_files()