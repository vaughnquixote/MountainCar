# Mountain Car
This repo houses a set of python scripts which provide multiple solutions to the classic mountain car problem in reinforcement learning. I chose to work with the mountain car environment provided by OpenAi Gym. The basic task is for a small car to drive itself up a steep hill. The car begins in a valley and does not generate enough force to drive directly to the top of the hill. The only way for it to reach the top is by driving back and forth, leveraging its potential energy and applying force in the right moments.

I completed this as an indpendent project as a part of the Foundations of AI (CS 5100) course at Northeastern University while pursuing a master's degree in computer science.

## Project Structure

- `project_report.pdf`: final report prepared for Foundations of AI (CS 5100) at Northeastern. This document contains a thorough description of the problem, the algorithms used to solve the problem and the results. 
- `mountaincardriver.py`: driver script used to run the tabular mountain car agent. not integrated with the semi-gradient mountain car agent to be configurable
- `semi_gradient_mc.py`: the implementation of the semi-gradient SARSA mountain car agent implemented with a tile coding strategy for feature construction
- `tabular_mc.py`: the implementation of a tabular q-learning agent for mountain car
- `tiling.py`: the set of functions used for tile coding the features for the semi-gradient SARSA agent. this was implemented by Richard Sutton and made available through his website (http://incompleteideas.net/tiles/tiles3.html). I discovered this in Sutton and Barton's fabulous book *Reinforcement Learning: An Introduction*
- `mcdataviz.py`: script used to produce the data visualizations using matplotlib.pyplot. this needs to be
dropped into the directory containing the generated training data in order to be used. 