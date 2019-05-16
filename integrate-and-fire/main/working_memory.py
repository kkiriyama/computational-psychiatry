import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neuron import Neuron
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_NEURONS = 300
NUM_PRESTEPS = 10
NUM_STEPS = 100
V_ACT = 50


class WorkingMemory:
    def __init__(self):
        self.N = NUM_NEURONS
        self.matrix = self.randomNetworkGenerator()
        self.neurons = [Neuron(len(self.matrix[j, :])) for j in range(self.N)]
        self.firing_rate = []

    def randomNetworkGenerator(self):
        return np.uint8(np.random.rand(self.N, self.N) > 0.95)

    def loop(self, steps):
        for _ in tqdm(range(steps)):
            new_state_neurons = deepcopy(self.neurons)
            num_firing_neurons = 0
            for i, _ in enumerate(self.neurons):
                k_list = [0] * NUM_NEURONS
                for j, k in enumerate(self.matrix[:, i]):
                    if k == 1 and self.neurons[j].V >= V_ACT:
                        k_list[j] = 1
                new_state_neurons[i].currentUpdate(k_list)
                new_state_neurons[i].voltageUpdate()
                if new_state_neurons[i].V >= V_ACT:
                    num_firing_neurons += 1
            self.firing_rate.append(num_firing_neurons)
            self.neurons = deepcopy(new_state_neurons)

    def collective_stimulate(self, num_stimulate_neurons, stimulate_mA, stimulateDurations):
        for i in range(num_stimulate_neurons):
            self.neurons[i].stimulate(stimulate_mA, stimulateDurations)
    

wm = WorkingMemory()
wm.loop(NUM_PRESTEPS)
wm.collective_stimulate(2, 10.1, 10)
wm.loop(NUM_STEPS)

plt.plot(wm.firing_rate)
plt.show()
