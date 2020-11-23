import gym 
import numpy as np 
import random

GAMMA = 0.85
ALPHA = 0.4


class MountainCarAgent():

	def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, episodes=100):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.q_values = dict()
		self.env = gym.make('MountainCarAgent-v0')
		self.episodes = episodes


	def getQValue(self, position, velocity, action):

		value = self.q_values[(position, velocity, action)]

		if value == None:
			value = 0

		return value

	def train(self):

		state = env.reset()

		prev_position = round(state[0], 2)

		prev_velocity = round(state[1], 2)

		while done != True:

			if random.random() > self.epsilon:

				best_action = None

				max_q = float('-inf')

				for action in range(3):

					next_q = self.getQValue(prev_position, prev_velocity, action)

					if next_q > max_q:

						best_action = action

						max_q = next_q

			else:
				
				action = env.action_space.sample()



			observation, reward, done, info = env.step(action)

			position = round(observation[0], 2)

			velocity = round(observation[1], 2)

			current_q = q_values.get(position, velocity, action)

			qs = []

			for action in range(3): 

				qs.append(self.getQValue(prev_position, prev_velocity, action))

			next_q = max(qs)

			q_values[(position, velocity, action)] =  current_q 
						+ self.alpha * (reward + self.discount * next_q  - current_q)

			prev_position = position

			prev_velocity = velocity

	def getValueFromQValue(self):
		print("not defined")

	def getAction(self):
		print("not defined")


def main():

	print("bado, bado sana")

main()