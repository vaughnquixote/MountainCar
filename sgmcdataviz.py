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

def plot_first_win_vs_alpha(alphas, fw02, fw04, fw06, fw08):

	colors = ("black","red", "green", "blue")
	# data = (fw025, fw05, fw075, fw1)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(alphas, fw08, color="k", label="epsilon=0.8")
	ax.scatter(alphas, fw02, color="r", label="epsilon=0.2")
	ax.scatter(alphas, fw04, color="g", label="epsilon=0.4")
	ax.scatter(alphas, fw06, color="b", label="epsilon=0.6")

	z1 = np.polyfit(alphas, fw08, 1)
	p1 = np.poly1d(z1)
	plt.plot(alphas, p1(alphas), color="k")

	z2 = np.polyfit(alphas, fw02, 1)
	p2 = np.poly1d(z2)
	plt.plot(alphas, p2(alphas), color="r")

	z3 = np.polyfit(alphas, fw04, 1)
	p3 = np.poly1d(z3)
	plt.plot(alphas, p3(alphas), color="g")

	z4 = np.polyfit(alphas, fw06, 1)
	p4 = np.poly1d(z4)
	plt.plot(alphas, p4(alphas), color="b")

	plt.title('First Win vs Learning Rate')
	plt.xlabel('Alpha')
	plt.ylabel('First Win')
	plt.legend(loc=2)
	plt.show()

def plot_avg_reward_vs_alpha(alphas, aw02, aw04, aw06, aw08):

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(alphas, aw08, color="k", label="epsilon=0.8")
	ax.plot(alphas, aw06, color="r", label="epsilon=0.6")
	ax.plot(alphas, aw04, color="g", label="epsilon=0.4")
	ax.plot(alphas, aw02, color="b", label="epsilon=0.2")

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

	episodes = [100*(i+1) for i in range(10)]

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
		if(start == 1000):
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

	filename = "SGtrainingepisodes_"

	trained = "SGtrained_"

	alpha = 1

	trainingepisodes = dict()

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

	# plot_first_win_vs_alpha(alphas, first_wins_02, first_wins_04, first_wins_06, first_wins_08)

	# plot_avg_reward_vs_alpha(alphas, aw_02, aw_04, aw_06, aw_08)


extract_data_from_files()