import gym 

GAMMA = 0.85
ALPHA = 0.4

def main():

	q_values = dict()

	env = gym.make('MountainCar-v0')

	print(env.action_space)

	env.reset()

	for _ in range(10):

		env.render()
		action = env.action_space.sample()
		# actions are 0, 1 or 2 corresponding to push left, no push and push right
		observation, reward, done, info = env.step(action)
		print(observation)


	env.close()

