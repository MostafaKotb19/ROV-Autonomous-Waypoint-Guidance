
# ROV-Autonomous-Navigation
Controlling an ROV to do simple navigation and obstacle avoidance tasks using Deep Reinforcement Learning and PPO (Proximal Policy Optimization).

The repository contains manual controlling to test the environment and adjust rewards.
It also contains the architecture and design of the PPO Agent and the control configurations to train the model.

## Install
It is recommended to create a conda environment or a virtual environment.
To create a conda environment:
```
conda create -n myenv
conda activate myenv
```
To create a virtual environment:
```
python -m venv venv
source venv/bin/activate
```
After creating the environment, install the requirements:
```
pip install -r requirements.txt
```

## How It works
### Manual Control
[Main.py](manual_control/Main.py) is the main executable. You can use it to test the environment and enjoy manually controlling the ROV to complete the game.

[KeyboardController.py](manual_control/KeyboardController.py) Initializes the KeyboardController. It handles the conversion of the pressed keys into commands for the ROV thrusters.

[scenario.py](manual_control/scenario.py) contains the scenario and agent configurations, like world, agent_type, sensors, etc.

[CustomEnvironment.py](manual_control/CustomEnvironment.py) makes the environment from the scenario file. In addition, it adds the targets and obstacles, handles target choosing, and updates states.

[RewardFunction.py](manual_control/RewardFunction.py) contains the calculations for the reward function. It also contains definitions for some scenarios, like having collisions, getting outside the box, reaching a target, staying static, etc.
### PPO
[Main.py](PPO/Main.py) is the main executable. It contains the training of the PPO Model and the visualization of the training process.

[scenario.py](PPO/scenario.py) contains the scenario and agent configurations, like world, agent_type, sensors, etc.

[CustomEnvironment.py](PPO/CustomEnvironment.py) makes the environment from the scenario file. In addition, it adds the targets and obstacles, handles target choosing, and updates states.

[RewardFunction.py](PPO/RewardFunction.py) contains the calculations for the reward function. It also contains definitions for some scenarios, like having collisions, getting outside the box, reaching a target, staying static, etc.

[PPOAgent.py](PPO/PPOAgent.py) configures the PPO policy and value networks' architectures. It handles updating networks, saving and loading model, and logging losses per episode.

## Further Developing
For further developing, please visit HoloOcean Documentation:
[https://holoocean.readthedocs.io/en/latest/index.html](https://holoocean.readthedocs.io/en/latest/index.html)

It contains all the needed instruction to edit the agent, obstacles, and/or targets, create custom scenarios, add sensors, etc.
