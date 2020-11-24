import gym 
import numpy as np 
import random
import time

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
		print(self.env.observation_space.low)
		print(self.env.observation_space.high)


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
		random_action_counter = 0
		non_random_counter = 0

		while done != True:

			if num_episode > self.episodes - 5:
				self.env.render()

			if random.random() > self.epsilon:
				action = None
				max_q = float('-inf')

				for a in range(3):
					samp_q = self.q_values[(prev_position, prev_velocity, a)]
					if samp_q > max_q:
						action = a
						max_q = samp_q
				non_random_counter += 1
			else:			
				action = self.env.action_space.sample()
				random_action_counter += 1

			observation, reward, done, info = self.env.step(action)
			position = round(observation[0], 1)
			velocity = round(observation[1], 2)
			
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

		return total_reward, random_action_counter, non_random_counter

	def train(self):

		randoms = 0

		non_randoms = 0

		done_pos = False

		epsilon_change = self.epsilon/self.episodes

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


		for _ in range(self.episodes):	

			reward, rand, non_rand = self.run_episode(_)

			randoms += rand

			non_randoms += non_rand

			self.rewards.append(reward)

			if self.epsilon > 0.001:

				self.epsilon -= epsilon_change

			# if (_ % 100) == 0:

			# 	print("Episode: ", _)
			# 	print("Average of last 100 rewards: ", sum(self.rewards[-100:-1])/100)


		# for key, value in self.q_values.items():
		# 	print("q value: ", key, value)

		self.env.close()

	def get_rewards(self):

		return self.rewards

def main():
	start = time.time()
	mc_test = MountainCarAgent(0.2, 0.9, 0.8, 5000)

	mc_test.train()
	end = time.time()
	print("time taken: ", end - start)
	print("Maximum reward achieved: ", max(mc_test.get_rewards()))
	print("Last ten rewards achieved: ", mc_test.get_rewards()[-10:-1])

main()