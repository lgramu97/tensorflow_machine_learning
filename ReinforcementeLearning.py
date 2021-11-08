'''
    Q-Learning example. Open AI Gym.
'''

import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def get_average(values):
    return sum(values)/len(values)

def plot_results(rewards):
    # we can plot the training progress and see how the agent improved
    avg_rewards = []
    for i in range(0, len(rewards), 100):
        avg_rewards.append(get_average(rewards[i:i+100])) 

    plt.plot(avg_rewards)
    plt.ylabel('average reward')
    plt.xlabel('episodes (100\'s)')
    plt.show()
  

def run():
    # we are going to use the FrozenLake enviornment
    env = gym.make('FrozenLake-v1') 
    STATES = env.observation_space.n
    ACTIONS = env.action_space.n
    Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values 

    #Define some hyperparameters.
    EPISODES = 20000 # how many times to run the enviornment from the beginning
    MAX_STEPS = 10  # max number of steps allowed for each run of enviornment
    LEARNING_RATE = 0.81  # learning rate
    GAMMA = 0.96
    RENDER = True # if you want to see training set to true

    epsilon = 0.9

    '''
        Function we can use. We must go from S to G, and dont fall in H.

        S: initial state
        F: frozen lake
        H: hole
        G: the goal
        Red square: indicates the current position of the player
    '''
    print("Action space: ", env.action_space)# get number of actions (Action)
    print("Observation space: ", env.observation_space)  # get number of states (Enviroment)
    env.reset()  # reset enviornment to default state
    action = env.action_space.sample()  # get a random action 
    new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
    env.render()   # render the GUI for the enviornment 

    rewards = []
    for episode in range(EPISODES):

        state = env.reset()
        for _ in range(MAX_STEPS):
            
            if RENDER:
                env.render()

             # code to pick action
            if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
                action = env.action_space.sample()  # take random action
            else:
                action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values

            next_state, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

            if done: 
                rewards.append(reward)
                epsilon -= 0.001
                break  # reached goal
        
    print(Q)
    print(f"Average reward: {sum(rewards)/len(rewards)}:")
    # and now we can see our Q values!

    plot_results(rewards)



if __name__ == '__main__':
    run()