This environment is similar as the previous 'Reaching task' folder, but which is compatiable with gym environment.
I have already used 'stable-baselines' tested this environment, it can work well. You can also take a look at 'DDPG+HER_not_official_version', which include some DDPG+HER implementation that use mpi. 

2 Important files: 1. ur5_symbol_v0.py : This environment is compatiable for most of algorithms which can be applied to continous action space.
                  2. ur5_symbol_v1.py : This is environment is designed for HER like algorithms, the observation space is in dict format.
                  
For the current UR5 gym environment for reaching task, it still has some drawbacks to be improved, if I got time, I will fix it, which are the following:
1. Doesn't include render part. Actually I am currently using some sentense "I am reaching the target!!!!!!!!!!!!!!", "Touching b3"... to observe the real time performance.
2. Add a config.json file to store all configuration of the environment. Current code is messy.
3. Write v0 and v1 together to make the envrionment be compatiable with HER and PPO and other algorithms.
4. These moving limit of robot's joint still need to be re-colibarated.
5. Need to have another robot's body checking system, in order that robot's body won't touch safety boundary and self-collision. As to the current one, it can only gurrantee that tip won't touch the boundary.
6. Check_done function still need to be improved.


TODOs: 1. Create another environment, which will take picture as observation.
       2. Create another task, maybe pick and place.
