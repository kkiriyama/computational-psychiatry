import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


constants = {
    'conductance': {
        'Na_max': 120,
        'K_max': 36,
        'leak': 0.3 
    },
    'equation_voltage': {
        'E_Na': 50,
        'E_K':  -77,
        'E_leak': -54.387
    },
    'capacity': 1
}


class HodgkinHuxley:
    def __init__(self):
        self.V = -65
        self.I_Na = 0
        self.I_K = 0
        self.I_leak = 0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.stimulateDuration = 0
        self.t = sp.arange(0.0, 100.0, 0.01)

    def alpha_m(self, V):
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
    
    def beta_m(self, V):
        return 4.0*sp.exp(-(V+65.0) / 18.0)
    
    def alpha_h(self, V):
        return 0.07*sp.exp(-(V+65.0) / 20.0)
    
    def beta_h(self, V):
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
    
    def alpha_n(self, V):
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        return 0.125*sp.exp(-(V+65) / 80.0)
    
    @staticmethod
    def step(X, t, self):
        V, m, h, n = X
        dm = self.alpha_m(V) * (1-m) - self.beta_m(V) * m
        dn = self.alpha_n(V) * (1-n) - self.beta_n(V) * n
        dh = self.alpha_h(V) * (1-h) - self.beta_h(V) * h

        dV = -1 / constants['capacity'] * (
                        constants['conductance']['Na_max'] * m**3 * h * (V - constants['equation_voltage']['E_Na']) +
                        constants['conductance']['K_max'] * n ** 4 * (V - constants['equation_voltage']['E_K']) +
                        constants['conductance']['leak'] * (V - constants['equation_voltage']['E_leak']) -
                        self.stimulateCurrent(t)
        )
        return dV, dm, dh, dn
    
    def stimulateCurrent(self, t):
        return 10 * (t > 20) - 10 * (t > 80)

    def main(self):
        X = odeint(self.step, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))

        plt.plot(X[:, 0])
        plt.show()

HH = HodgkinHuxley()
HH.main()
