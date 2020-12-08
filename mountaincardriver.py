import gym 
import numpy as np 
import random
import time
import matplotlib.pyplot as plt

GAMMA = 0.85
ALPHA = 0.4


class MountainCarAgent():

	def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, episodes=100):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.q_values = dict()
		self.env = gym.make('MountainCar-v0')
		self.episodes = episodes
		self.rewards = []
		self.winning_episodes = []
		self.final_positions = []


	def getQValue(self, position, velocity, action):

		value = self.q_values.get((position, velocity, action))

		if value == None:
			value = 0

		return value

	def run_episode(self, num_episode):

		state = self.env.reset()
		prev_position = round(state[0], 1)
		prev_velocity = round(state[1], 2)
		total_reward = 0
		done = False

		while done != True:

			# if num_episode > self.episodes - 5 or num_episode < 5:
			# 	self.env.render()


			if random.random() > self.epsilon:
				action = None
				max_q = float('-inf')

				for a in range(3):
					samp_q = self.q_values[(prev_position, prev_velocity, a)]
					if samp_q > max_q:
						action = a
						max_q = samp_q
			else:			
				action = self.env.action_space.sample()

			observation, reward, done, info = self.env.step(action)

			position = round(observation[0], 1)
			velocity = round(observation[1], 2)

			if done and position >= 0.5:
				self.q_values[(prev_position, prev_velocity, action)] = reward
				total_reward += reward
				break

			current_q = self.q_values[(prev_position, prev_velocity, action)]
			
			qs = []

			for act in range(3): 
				qs.append(self.q_values[(position, velocity, act)])
			next_q = max(qs)

			
			self.q_values[(prev_position, prev_velocity, action)] =  current_q \
						+ self.alpha * (reward + self.gamma * next_q  - current_q)

			prev_position = position
			prev_velocity = velocity 
			total_reward += reward

		if done == True and position >= 0.5:

			self.wins.append(num_episode)

		self.final_positions.append(position)

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

			# if(_ < 5 or _ > self.episodes - 5):
			# 	time.sleep(10)


		# plt.plot(episodes, self.rewards)
		# plt.title("Test plot 1")
		# plt.xlabel("Episode")
		# plt.ylabel("Reward")
		# plt.show()


		self.env.close()

	def initialize_q_values(self):

		done_pos = False

		pos_count = -1.2

		while not done_pos:

			done_vel = False

			vel_count = -0.07

			while not done_vel:

				for i in range(3):

					self.q_values[(pos_count, vel_count, i)] = random.uniform(-1, 1)

				vel_count += 0.01

				vel_count = round(vel_count, 2)

				if vel_count > 0.07:

					done_vel = True

			pos_count += 0.1

			pos_count = round(pos_count, 1)

			if pos_count > 0.6:

				done_pos = True

	def get_rewards(self):

		return self.rewards

	def set_epsilon(self, new_ep):

		self.epsilon = new_ep

	def set_num_episodes(self, num_episodes):

		self.num_episodes = num_episodes

	def get_winning_episodes(self):

		return self.winning_episodes


def main():
	mc_test = MountainCarAgent(0.2, 0.9, 0.8, 10000)
	mc_test.initialize_q_values()
	mc_test.train()

	wins = mc_test.get_winning_episodes()

	alpha_variation = False
	epsilon_variation = False
	alpha = 0.05
	epsilon = 0.05
	while not alpha_variation: 
		# alpha, gamma, epsilon, episodes
		while not epsilon_variation:
			mc = MountainCarAgent(alpha, 0.9, epsilon, 10000)
			mc.initialize_q_values()
			mc.train()

			wins = mc.get_winning_episodes()
			first_win = wins[0]

			








	# mc_test.set_epsilon(0)
	# mc_test.set_num_episodes(100)
	# mc_test.train()



	print("time taken: ", end - start)
	print("Maximum reward achieved: ", max(mc_test.get_rewards()))
	print("Last ten rewards achieved: ", mc_test.get_rewards()[-10:-1])
	print("Average reward achieved: ", sum(mc_test.get_rewards())/len(mc_test.get_rewards()))

main()