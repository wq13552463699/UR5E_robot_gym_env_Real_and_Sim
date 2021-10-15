# UR5 Robotic arm GYM simulation environment(CoppeliaSim + Pixel obervation + Reaching task)
Author: Qiang Wang
This combination of environment + simulator is not like classic simulators, such as pyrep and pybullet, which can be trained by direct connection, that is, no visual training. During the training process in this environment, coppeliaSim runs in real time, and the disadvantage is that the training speed is slow.

## Installation
Please check the packages in requirement_reach.txt and install the packages you don’t have

# Environment setup
The environments I was using was included in two .ttt files, One of the environments has more input, and you can choose at most three cameras, while the other environment has less input, only one camera input. as follows:
<img src="https://github.com/wq13552463699/UR5E_robot_gym_env_Real_and_Sim/blob/main/Simulation/images/2.png" width="633" >\
<img src="https://github.com/wq13552463699/UR5E_robot_gym_env_Real_and_Sim/blob/main/Simulation/images/3.png" width="633" >\

## Implementation
The CoppeliaSim_UR5_gym_v0.py and CoppeliaSim_UR5_gym_v3.py files are the code of the environment you want to use. \
Its format is gym format. You can use any algorithms code that is compatible with the gym environment.\
You can configure the parameters in the environment by changing the code, such as: 1. The size of the input image; 2. The reward function, etc.

## Detailed explanation of the contents of the files in the folder:
CoppeliaSim_env.py: Control the connection with CoppeliaSim, the bottom communication operation.\
CoppeliaSim_UR5_base.py: Control the movement of the UR5 robot in CoppeliaSim, read the state value of the robot, and build on CoppeliaSim_env.\
rl_utils: Some functions needed\
kinematic.py: Forward kinematic and Inverse Kinematic are calculated here\
math_cal.py: some useful math methods for calculating transfer matrix, pos, quardranion\
reach_task.ttt: coppeliasim environment file\
remoteApi.so/sim.py/simConst.py: Leave it here, these files are important for communicating with coppeliasim.These are official files, you can go into these files to see what they are doing\
rl_symbol_env_continous: The mathematical continuous environment\
rl_symbol_env: The mathematical sparse environment\
ur5_kinematic.py: kinematics that compatible with the environment in reach.ttt\
requirement_reach.txt: all packages I have in my conda environment\
folder "Reaching_task" and "reaching_gym" :The code used in the past, including gym and non-gym reinforcement learning environments and algorithms, has not been sorted out, is messy, and has stopped updating. Here just for the purpose of copying





























