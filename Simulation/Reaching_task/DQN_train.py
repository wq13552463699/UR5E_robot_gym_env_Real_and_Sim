from collections import deque
import tensorflow as tf
from DQN_agent import DQNAgent
import numpy as np
import random
from rl_symbol_env import symbol_env

# setting seeds for result reproducibility. This is not super important
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)

# Hyperparameters / Constants
EPISODES = 500000

REPLAY_MEMORY_SIZE = 1000000
#Increased

MINIMUM_REPLAY_MEMORY = 1000
MINIBATCH_SIZE = 128

EPSILON = 0
EPSILON_DECAY = 0.999
MINIMUM_EPSILON = 0.05
MINIMUM_EPSILON_stage_1 = 0.95
MINIMUM_EPSILON_stage_2 = 0.05
DISCOUNT = 0.8
AWARD_RATIO = 1000
PUNISH_RATIO = -30



# Environment details
env = symbol_env()
# env = gym.make(ENV_NAME)
action_dim = 10
observation_dim = (12,)

# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.Session()

# Replay memory to store experiances of the model with the environment
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# Our models to solve the mountaincar problem.
agent = DQNAgent(sess, action_dim, observation_dim)
agent.model.load_weights('/home/qiang/Pro/10000_agent_.h5')


def train_dqn_agent():
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X_cur_states = []
    X_next_states = []
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        X_cur_states.append(cur_state)
        X_next_states.append(next_state)

    X_cur_states = np.array(X_cur_states)
    X_next_states = np.array(X_next_states)

    # action values for the current_states
    cur_action_values = agent.model.predict(X_cur_states)
    # action values for the next_states taken from our agent (Q network)
    next_action_values = agent.model.predict(X_next_states)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if not done:
            # Q(st, at) = rt + DISCOUNT * max(Q(s(t+1), a(t+1)))
            cur_action_values[index][action] = reward + DISCOUNT * np.amax(next_action_values[index])
        else:
            # Q(st, at) = rt
            cur_action_values[index][action] = reward
    # train the agent with new Q values for the states and the actions
    agent.model.fit(X_cur_states, cur_action_values, verbose=0)

global_success_time = 0
success_time = 0
max_reward = -999999
step = 0
done = 1
history = {"Episode":[],"Success_rate":[]}

history["Episode"].append(1)
for episode in range(EPISODES):
    if done == 1:
        cur_state = env.reset()
    if done == 2:
        cur_state = env.reset_o()
    # print("Target pos:", env.target_pos)
    done = 0
    episode_reward = 0
    episode_length = 0
    step += 1
    while not done:
        episode_length += 1
        # step += 1
        # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower.
        # if VISUALIZATION:
        #     env.render()

        if (np.random.uniform(0, 1) < EPSILON):
            # Take random action
            action = np.random.randint(0, action_dim)
            # print(action)
        else:
            # Take action that maximizes the total reward
            action = np.argmax(agent.model.predict(np.expand_dims(cur_state, axis=0))[0])

        # print("Action I take", action)
        next_state, reward, done = env.step(action)
        # if episode_length > 2000:
        print(cur_state,action, reward)
        # print("Current state", next_state)
        # print("Reward I got", reward)
        # print("done", done)
        # print(done)

        # print(done)
        # if reward == 0:
        #     done = 2
        # print(reward)
        if done == 1:
            # If episode is ended the we have won the game. So, give some large positive reward
            reward = AWARD_RATIO
            success_time += 1
            global_success_time += 1
        if done == 2:
            reward = PUNISH_RATIO

        episode_reward += reward
        #     # save the model if we are getting maximum score this time
        #     if (episode_reward > max_reward):
        #         agent.model.save_weights(str(episode_reward) + "_agent_.h5")
        #
        # elif done == 2:
        #     reward = episode_reward - 250
        #     if (episode_reward > max_reward):
        #         agent.model.save_weights(str(episode_reward) + "_agent_.h5")

        # else:
        #     # In oher cases reward will be proportional to the distance that car has travelled
        #     # from it's previous location + velocity of the car
        #     reward = 5 * abs(next_state[0] - cur_state[0]) + 3 * abs(cur_state[1])

        # Add experience to replay memory buffer
        replay_memory.append((cur_state, action, reward, next_state, done))
        cur_state = next_state
        
        
        if (len(replay_memory) > MINIMUM_REPLAY_MEMORY):
            train_dqn_agent()
        

    if (len(replay_memory) < MINIMUM_REPLAY_MEMORY):
        continue
        
        

        
        # print("Training")

    if (EPSILON > MINIMUM_EPSILON and len(replay_memory) > MINIMUM_REPLAY_MEMORY):
        EPSILON *= EPSILON_DECAY
        
    # if (episode_length > 100000 and EPSILON > MINIMUM_EPSILON_stage_2):
    #     EPSILON *= EPSILON_DECAY
        
    # some bookkeeping.
    avg_reward = episode_reward / episode_length
    
    # if (episode % 10) == 0 and episode != 0:
    #     train_dqn_agent()
    
    if (episode % 500) == 0 and episode != 0:
        agent.model.save_weights(str(episode) + "_agent_.h5")
    
    # if (avg_reward > max_reward):
    #     agent.model.save_weights(str(episode_reward) + "_agent_.h5")
    max_reward = max(avg_reward, max_reward)

    print('Episode', episode,'| ', 'Episodic Reward', episode_reward,'| ', 'Maximum Reward', max_reward, '| ','End_step',episode_length, '| ','EPSILON', EPSILON)
    print('-------------------------------------------------------------------------------------------------------' )

    if episode % 500 == 0:
        print("For the last 500 episodes, I successfully reach the target ",success_time," times")
        print("For the last ",episode," episodes", ", I successfully reach the target ",global_success_time,"times")
        print("Success rate: ",global_success_time / step)
        success_time = 0