import gym 
import matplotlib.pyplot as plt
import csv
from tabular_mc import TabularMountainCarAgent


def main():

	alpha_variation = False
	epsilon_variation = False
	alpha = 0.1

	fileextension = ".csv"

	filename = "trainingepisodes_"

	trained = "trained_"

	first_win = -1

	while alpha <= 1: 

		epsilon = 0.25

		while epsilon <= 1:
            # train the agent across 5000 episodes with the given alpha 
			# and epislon values and a gamma value of 0.9
			mc = TabularMountainCarAgent(alpha, 0.9, epsilon, 5000)
			mc.initialize_q_values()
			mc.train()

			wins = mc.get_winning_episodes()

			if(len(wins)>0):
				first_win = wins[0]

			mc.set_epsilon(0)
			mc.set_num_episodes(100)
			mc.train()

			final_positions = mc.get_final_positions()[:-100]
			rewards = mc.get_rewards()[:-100]
			max_positions = mc.get_max_positions()[:-100]

			trained_rewards = mc.get_rewards()[-100:]
			trained_final_positions = mc.get_final_positions()[-100:]
			trained_max_positions = mc.get_max_positions()[-100:]

			alpha_str_rep = str(alpha).replace(".", "")
			epsilon_str_rep = str(epsilon).replace(".", "")

			full_file_name = filename + str(alpha_str_rep) + "_" + str(epsilon_str_rep) + fileextension
			with open(full_file_name, mode="w") as file:
				writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
				writer.writerow(["Episode", "TotalReward", "FinalPosition", "MaxPosition"])
				if(first_win >= 0):
					writer.writerow(["Episode of first win: " + str(first_win)])
				for i in range(len(rewards)):
					writer.writerow([i, rewards[i], final_positions[i], max_positions[i]])

			trained_file = trained + str(alpha_str_rep) + "_" + str(epsilon_str_rep) + fileextension
			with open(trained_file, mode="w") as file2:
				writer = csv.writer(file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
				writer.writerow(["Episode", "Reward, FinalPosition", "MaxPosition"])
				for i in range(len(trained_rewards)):
					writer.writerow([i, trained_rewards[i], trained_final_positions[i],trained_max_positions[i]])

			epsilon += 0.25

		alpha += 0.1

main()