import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neuron import Neuron
import matplotlib.pyplot as plt

neuron = Neuron(0)

voltage = []

for i in range(20):
    voltage.append(neuron.V)
    neuron.voltageUpdate()
    neuron.currentUpdate([])

neuron.stimulate(mA=10.5, duration=10)

for i in range(100):
    voltage.append(neuron.V)
    neuron.voltageUpdate()
    neuron.currentUpdate([])

plt.plot(voltage)
plt.show()