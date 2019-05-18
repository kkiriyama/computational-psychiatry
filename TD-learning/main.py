import matplotlib.pyplot as plt

def stimulate(t):
    if t == 10:
        return 1
    return 0

def reward(t):
    if t == 90:
        return 1
    return 0

class TDLearning():
    def __init__(self):
        self.gamma = 1.0
        self.value = []
        self.S = []
        self.weight = [0] * 100
        self.time = 100
        self.lr = 0.1
    
    def forward(self):
        self.value = []
        for t in range(self.time):
            self.S.append(stimulate(t))
            V = sum([self.weight[tau] * self.S[t - tau] for tau in range(t)])
            self.value.append(V)
    
    def backward(self):
        deltas = []
        for t in range(self.time - 1):
            delta = reward(t) + self.gamma * self.value[t+1] - self.value[t]
            deltas.append(delta)
            for tau in range(t):
                self.weight[tau] += self.lr * delta * stimulate(t - tau)
        return deltas
    
    def main(self):
        TDs = []
        for i in range(1000):
            self.forward()
            deltas = self.backward()
            TDs.append(deltas)
        
        for i in range(1, 1000, 200):
            plt.plot(TDs[i], label="iteration=%d" % (i))
        plt.legend()
        plt.show()

TDL = TDLearning()
TDL.main()
