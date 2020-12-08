import numpy as np 
import random
import matplotlib.pyplot as plt
import csv


def extract_data_from_files():

	fileextension = ".csv"

	filename = "trainingepisodes_"

	trained = "trained_"

	while alpha <= 10: 

		alpha_str_rep = "0" + str(alpha)
		# alpha, gamma, epsilon, episodes
		epsilon = 25
		while epsilon <= 1:

			if(abs(epsilon - 1) < 0.001):
				epsilon_str_rep = "10"
			elif(abs(epsilon - 0.5) < 0.001):
				epsilon_str_rep
			elif(abs(epsilon - 0.5) < 0.001))

			full_file_name = filename + str(alpha_str_rep) + "_" + str(epsilon_str_rep) + fileextension
			with open(full_file_name, mode="r") as file:
				reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
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