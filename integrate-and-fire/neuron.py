V_LEAK = -70
V_THR = -50
V_ACT = 50
V_RESET = -55
C_M = 0.5
G_LEAK = 25
T_REF = 5
TAU = 100
EQ_VOLTAGE = 0

class Neuron:
    def __init__(self, N):
        self.V = V_LEAK
        self.restTime = 0
        self.synapse = [0]*N
        self.N = N
        self.I = 0
        self.sitimulateDuration = 0
        self.stimulation_mA = 0
    
    def currentUpdate(self, k_list):
        for j in range(self.N):
            ds = -1 * self.synapse[j] / TAU + k_list[j]
            self.synapse[j] += ds
        self.I = sum(self.synapse) * (self.V - EQ_VOLTAGE)

        if self.sitimulateDuration > 0:
            self.I += self.stimulation_mA
            self.sitimulateDuration -= 1


    def voltageUpdate(self):
        if self.restTime > 0:
            self.restTime -= 1
        else:
            if self.V < V_THR:
                dV = -1 / 100 * G_LEAK / C_M * (self.V - V_LEAK) + self.I
                if self.V + dV >= V_THR:
                    self.V = V_ACT
                else:
                    self.V += dV
            elif self.V >= V_THR:
                self.V = V_RESET
                self.restTime = T_REF
    
    def stimulate(self, mA, duration):
        self.sitimulateDuration = duration
        self.stimulation_mA = mA
        