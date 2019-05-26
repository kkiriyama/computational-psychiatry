import numpy as np

PATH_GRAPH = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5],
    4: [5],
    5: [0]
}

class Agent:

    def __init__(self, rewards, softmax_m):
        self.s = 0
        self.t = 0
        self.f = 0.9
        self.gamma = np.random.rand()
        self.B = np.zeros((6, 6))
        self.V = np.zeros(6)
        self.rewards = rewards
        self.softmax_m = softmax_m
    
    def actionSelection(self):
        self.B = np.zeros((6, 6))
        for state in range(6):
            next_state_list = PATH_GRAPH[state]
            for next_state in next_state_list:
                self.B[state, next_state] = self.V[next_state] + rewards[next_state] - self.V[state]
        select_action_p = self.B[state, :] / np.sum(self.B[state, :])
        action = np.random.choice(list(range(6)), 1, p=select_action_p)
        take_action_p = 1 / (1 + np.exp(-self.softmax_m * (self.B[state, next_state]-1)))
        taken_action = np.random.choice([action, self.s], 1, p=take_action_p)
        return taken_action
    
    