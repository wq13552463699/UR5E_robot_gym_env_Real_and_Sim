# UR5e Robotic arm GYM reinforcement learning environment(Pixel obervation + Reaching task)
    Author: Qiang Wang
    
## Highlight:
1. The reset system of the UR5e robot arm, when the UR5E robot tends to cross the safety range, the UR5e robot arm will be reset.Forward_kin_v1.py defines the forward kinematics model of the UR5E robot, which allows the detailed point coordinates of the robot's body to be output when only the joint angle of the robot is input. For details, please see an of my folder: https://github.com/wq13552463699/UR5E_robot_advanced_forward_kinematic_system
2. Automatic rewarding system: Detect the position of the target object in the environment in the robot coordinates, and calculate the distance between the end effector of the robot and the target object at each step.\
    **Principle**: \
    	       1. Train the detectron2 neural network, the objects that can be detected include circles, balls, rectangles and cuboids.\
               2. Place 4 rectangular labels in the four corners of the environment as a reference.\
               3. Place an object in the environment and use the trained detectron to capture the object in the environment. And calculate the relative coordinates between the object and the label.\
               4. Transform the picture coordinates into robot coordinates through coordinate transformation.
               
## Robot control method: 
Currently only joint control is supported, please see the example video: https://www.youtube.com/watch?v=pjcWBtBDatA \
		This video does not represent the results of training, but is just an example of environmental work.

## Installaiton
1. Please check the packages in requirement_reach.txt and install the packages you don’t have
2. Super Important packages: detectron2; pyrealsense2; urx

## Tutorial for UR5e robot to connect with computer via TCP/IP:
The prerequest is to buy a sufficiently long network cable, one end is plugged into the robot's control box's externet cable interface, and the other end is plugged into the computer.
1. Activate the robot's TCP/IP connection. (You can skip this step because it is supposed to activate automatically every time the robot starts. But if the robot's history is cleared or an error happens, you need to check and activate it). To do it:
	a. Go to Manual mode(If you are in the Remote control mode, you need to go to Local control mode firstly.) Password for accessing manual mode: ****** Ask from people :)
	b. Go to setting > System > Network, what should be shown on this page is Static address, and the configuration of the Network detailed settings is IP address: 192.168.75.128, Subnet mask: 255.255.255.0
		Default gateway: 0.0.0.0  P-DNS: 0.0.0.0 A-DNS:0.0.0.0
	c. If you can not see the above information on this page, set it as above and click apply, then you should see Network is connected.

2. Configurate your Linux laptop(The language on my laptop is Chinese, so the translation may be a little different with your end)
	a. Go to ethernet connecting > settings > IPV4
	b. IPV4 method: Manual, it should be in DHCP firstly if you have never used it before, DON'T CHOOSE THAT.
	c. Input:  IP: 192.168.75.150   Subnet: 255.255.255.0   Gate:0.0.0.0  DNS:0.0.0.0
	d. Click Apply, then you should see information showing that the connecting is successful.

3. Install the API and run your code.

## Hardware setup
<img src="https://github.com/wq13552463699/UR5E_robot_gym_env_Real_and_Sim/blob/main/Real%20UR5e/images/42c2a3ff536ac961c121369f277d9c9.jpg" width="633" >\
	1. Universal robot UR5e robotic arm\
	2. Camera: Intel Real Sense D435\
	3. vention UR5E robotic arm working platform

## Implementation
The real_UR5_gym_v1.py file is the code of the environment you want to use. Its format is gym format. You can use any algorithms code that is compatible with the gym environment.
