import numpy as np

HEIGHT = 5
WIDTH = 5

class grid_world:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT


    def is_terminal(self, state):   # Goal state
        x, y = state
        return False


    def interaction(self, state, action):
        if self.is_terminal(state):
            return state, 0

        if state == [0,1]:
            next_state = [4,1]
            reward = 10
        elif state == [0,3]:
            next_state = [2,3]
            reward = 5
        else:
            next_state = (np.array(state) + action).tolist()
            reward = 0
            x, y = next_state
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                next_state = state
                reward = -1
        return next_state, reward


    def size(self):
        return self.width, self.height
