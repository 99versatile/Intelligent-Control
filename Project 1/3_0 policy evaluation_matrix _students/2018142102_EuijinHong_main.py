import numpy as np
from environment import grid_world
from visualize import draw_image

WORLD_SIZE = 5
# left, up, right, down
ACTIONS = {'LEFT':np.array([0, -1]), 'UP':np.array([-1, 0]), 'RIGHT':np.array([0, 1]), 'DOWN':np.array([1, 0])}
ACTION_PROB = 0.25



def evaluate_state_value_by_matrix_inversion(env, discount=0.9):
    WIDTH, HEIGHT = env.size()

    # Initializing Value function vector V to a numpy zero array with size 25
    V = np.zeros((25,), dtype=float)

    # Reward matrix R
    # Initializing Reward Matrix(vector) R to a numpy zero array with size 25
    R = np.zeros((25,), dtype=float) 
    # Transition matrix T 
    # Initializing Transition Matrix T to a numpy zero array with size 25 x 25
    T = np.zeros((25, 25), dtype=float) 
    # iterate over each state: S0 ~ S24 (25 states total)
    for i in range(WIDTH * HEIGHT):     
        # coordinate of the current state in the grid world; 
        # the indexing starts from left top and goes through the top-bottom direction)
        current_state = [i//5, i%5] 
        # iterate interaction for every action on the current state following the equiprobable policy     
        for action in ACTIONS.values():      
            # get the next_state(s') and reward(r) from interaction between agent and environment 
            # based on the current state and particular action
            next_state, reward = env.interaction(current_state, action)  
            # Add up the (reward x action_probability) for a particular action 
            # based on the current state and particular action
            R[current_state[0]*5+current_state[1]] += reward * ACTION_PROB
            # Add up the (action_probability x transition_probability) for a particular action 
            # based on the current state and particular action
            T[current_state[0]*5+current_state[1], next_state[0]*5+next_state[1]] +=  ACTION_PROB * 1 

    # Creating an Identity Matrix of size 25 x 25
    I = np.identity(25, dtype=float)

    # Evaluating the Value function V by calculating the Bellman Equation: ((I - gamma * T)^-1)R
    V = np.linalg.inv(np.subtract(I, discount * T)).dot(R)

    new_state_values = V.reshape(WIDTH, HEIGHT)
    draw_image(1, np.round(new_state_values, decimals=2))

    return new_state_values


if __name__ == '__main__':
    env = grid_world()
    values = evaluate_state_value_by_matrix_inversion(env = env)



