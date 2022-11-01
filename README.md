# Mountain Car
This repo houses a set of python scripts which provide multiple solutions to the classic mountain car problem in reinforcement learning. I chose to work with the mountain car environment provided by OpenAI Gym. The basic task is for a small car to drive itself up a steep hill. The car begins in a valley and does not generate enough force to drive directly to the top of the hill. The only way for it to reach the top is by driving back and forth, leveraging its potential energy and applying force in the right moments.

The car exists in a two dimensional environment. For any given state (position in the two dimensional environment), the car has the option of accelerating leftward, not accelerating or accelerating rightward. 

I completed this as an indpendent project as a part of the Foundations of AI (CS 5100) course at Northeastern University while pursuing a master's degree in computer science.

## Project Structure

- `project_report.pdf`: final report prepared for Foundations of AI (CS 5100) at Northeastern. This document contains a thorough description of the problem, the algorithms used to solve the problem and the results. 
- `mountaincardriver.py`: driver script used to run the tabular mountain car agent. not integrated with the semi-gradient mountain car agent to be configurable
- `semi_gradient_mc.py`: the implementation of the semi-gradient SARSA mountain car agent implemented with a tile coding strategy for feature construction
- `tabular_mc.py`: the implementation of a tabular q-learning agent for mountain car
- `tiling.py`: the set of functions used for tile coding the features for the semi-gradient SARSA agent. this was implemented by Richard Sutton and made available through his website (http://incompleteideas.net/tiles/tiles3.html). I discovered this in Sutton and Barton's fabulous book *Reinforcement Learning: An Introduction*
- `mcdataviz.py`: script used to produce the data visualizations using matplotlib.pyplot. this needs to be
dropped into the directory containing the generated training data in order to be used. 

## Approaches Used to Solve Mountain Car

### Tabular Q-Learning Agent

This agent is implemented in `tabular_mc.py`. A full description of the aglorithm can be found in `project_report.pdf`. 

The basic idea behind the tabular Q agent is that the state (two dimensional coordinate position) and action (encoded representation of the action taken) can be represented by a tuple of three numbers. The car then learns the approximate rewards (Q values) for these state-action pairs by training in the environment and stores them in a table (python dictionary). As the agent acts in the environment he rewards are observed and updated based upon the classic Q-Learning algorithm.

The trick to getting this approach to work is to dramatically reduce the size of the state space by rounding the floating point numbers representing the positional coordinates.

### Semi-Gradient SARSA Agent

This agent is implemented in `semi_gradient_mc.py`. A full description of the algorithm can be found in `project_report.pdf`. 

Rather than a simple table of values, this agent uses a function to approximate the value of a given state and action pair. The features used in the function were constructed using a tile coding strategy outlined in SUtton and Barto's *Reinforcement Learning: An Introduction*. Rather than the direct value of the approximate reward being updated based upon observations made by the agent, the weights of the features in the feature vector are updated. This agent performs much better than the tabular Q agent, both learning more quickly and learning a more effective strategy for the environment. This is likely because, in some sense, the tile-coding model is able to more accurately represent the mountain car environment.