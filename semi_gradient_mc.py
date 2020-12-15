import gym
import numpy as np 
import tiling
import csv
import random 

class SemiGradientMountainCarAgent():

	def __init__(self, num_tilings=8, tiling_dimension=8, alpha=0.5, gamma=0.9, epsilon=0.1, episodes=100):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = gym.make('MountainCar-v0')
		self.episodes = episodes
		self.rewards = []
		self.winning_episodes = []
		self.final_positions = []
		self.max_positions = []
		self.wins = []
		self.tile_hash = tiling.IHT(4096)
		self.weights = [0.0 for _ in range(4096)]
		self.trace_vector = [0 for _ in range(4096)]
		self.num_tilings = num_tilings
		self.tiling_dimension = tiling_dimension
		

	def getFeatures(self, position, velocity, action):
		# tiles(iht,8,[8*x/(0.5+1.2),8*xdot/(0.07+0.07)],A)
		features = tiling.tiles(self.tile_hash, self.num_tilings, [8*position/(0.5+1.2), 8*velocity/(0.07+0.07)], [action])
		return features

	def getQValues(self, position, velocity):
		features = [self.getFeatures(position, velocity, a) for a in range(3)]
		values = []
		for act_features in features:
			weighted_feature = [self.weights[f] for f in act_features]
			values.append(sum(weighted_feature))
		return values

	def getNextQ(self, position, velocity, action):
		features = self.getFeatures(position,velocity,action)
		values = [self.weights[feature] for feature in features]
		return values

	def updateQ(self, position, velocity, action, value):

		features = self.getFeatures(position,velocity,action)
		current = sum([self.weights[feature] for feature in features])
		delta = (value - current)
		for f in features:
			self.weights[f] += self.alpha * delta
		
	def run_episode(self, num_episode):

		state = self.env.reset()
		# prev_position = round(state[0], 1)
		# prev_velocity = round(state[1], 2)
		prev_position = state[0]
		prev_velocity = state[1]
		total_reward = 0
		done = False
		max_position = prev_position

		while done != True:

			if(num_episode > (self.episodes - 5)):
				self.env.render()

			if random.random() > self.epsilon:
				q_values = self.getQValues(prev_position, prev_velocity)
				action = q_values.index(max(q_values)) 
			else:			
				action = self.env.action_space.sample()

			observation, reward, done, info = self.env.step(action)

			# position = round(observation[0], 1)
			# velocity = round(observation[1], 2)
			position = observation[0]
			velocity = observation[1]

			if(position > max_position):
				max_position = position

			if done and position >= 0.5:
				self.updateQ(prev_position, prev_velocity, action, reward)
				total_reward += reward
				break

			qs = []
			
			for act in range(3): 
				qs.append(sum(self.getNextQ(position, velocity, act)))

			next_q = max(qs)
			
			value = reward + self.gamma * next_q

			self.updateQ(prev_position, prev_velocity, action, value)

			prev_position = position
			prev_velocity = velocity 
			total_reward += reward

		if done == True and position >= 0.5:

			self.wins.append(num_episode)

		self.final_positions.append(position)
		self.max_positions.append(max_position)

		return total_reward

	def train(self):

		episodes = []

		epsilon_change = self.epsilon/self.episodes

		for _ in range(self.episodes):	

			reward= self.run_episode(_)

			self.rewards.append(reward)

			episodes.append(_)

			if self.epsilon > 0.001:

				self.epsilon -= epsilon_change

		self.env.close()

	def get_rewards(self):

		return self.rewards

	def set_epsilon(self, new_ep):

		self.epsilon = new_ep

	def set_num_episodes(self, num_episodes):

		self.num_episodes = num_episodes

	def get_winning_episodes(self):

		return self.winning_episodes

	def get_final_positions(self):

		return self.final_positions

	def get_max_positions(self):

		return self.max_positions

def main():

	alpha_variation = False
	epsilon_variation = False
	alpha = 0.025

	fileextension = ".csv"

	filename = "SGtrainingepisodes_"

	trained = "SGtrained_"

	first_win = -1

	while alpha <= 0.125: 

		epsilon = 0.2

		while epsilon <= 0.8:

			mc = SemiGradientMountainCarAgent(alpha, epsilon, gamma = 0.9, episodes = 1000)
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
				# if(first_win >= 0):
				# 	writer.writerow(["Episode of first win: " + str(first_win)])
				for i in range(len(rewards)):
					writer.writerow([i, rewards[i], final_positions[i], max_positions[i]])

			trained_file = trained + str(alpha_str_rep) + "_" + str(epsilon_str_rep) + fileextension
			with open(trained_file, mode="w") as file2:
				writer = csv.writer(file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
				writer.writerow(["Episode", "Reward, FinalPosition", "MaxPosition"])
				for i in range(len(trained_rewards)):
					writer.writerow([i, trained_rewards[i], trained_final_positions[i],trained_max_positions[i]])

			epsilon += 0.2

		alpha += 0.025

main()