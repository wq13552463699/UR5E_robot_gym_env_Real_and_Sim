Reaching task:
You don't have to use pyrep, you can use coppeliasim directly.
For the current reaching task, we have 2 agents are available, which are DQN and DDPG.

Installtion:
Copy this directory localy and install packages you don't have in requirement_reach.txt(I am sorry that I copy my "conda list"
directly, you can take a look and check what you think may be needed.)

Introduction about the file included:
1. 4 .h5 files: the latest trained model, but not mature, just trained for 1 hour.
2. DDPG_agent.py: Include the DDPG_agent
3. DDPG_keras_main_reach.py: The main scrip to start the training by DDPG. 
4. DQN_agent.py: ...
5. DQN_train.py: The main scrip to start the training by DQN.
6. env_base.py: Coppeliasim's RL environment.
7. ir_utils: Some functions needed.
8. kinematic.py: Forward kinematic and Inverse Kinematic are calculated here.
9. math_cal.py: some useful math methods for calculating transfer matrix, pos, quardranion...
10. reach_task.ttt: coppeliasim environment file
11. remoteApi.so/sim.py/simConst.py: Leave it here, these files are important for communicating with coppeliasim.  
                                    These are official files, you can go into these files to see what they are doing.
12. rl_symbol_env_continous: The mathematical continuous environment.
13. rl_symbol_env: The mathematical sparse environment.
14. ur5_kinematic.py: kinematics that compatible with the environment in reach.ttt
15. UR5_get_env.py: not important.
16. requirement_reach.txt: all packages I have in my conda environment.

The above introduction may not be detailed, if you have questions, let me know.