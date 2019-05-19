import numpy as np
import matplotlib.pyplot as plt

VAR_D = 2
VAR_Y = 25

X = [0]
Y = [0]
pred_mean = [0]
pred_var = [0]

for t in range(100):
    x = np.random.normal(loc=X[-1], scale=np.sqrt(VAR_D), size=(1,))[0]
    X.append(x)
    y = np.random.normal(x, np.sqrt(VAR_Y), (1,))[0]
    Y.append(y)
    k = pred_var[-1] / (pred_var[-1] + 5)
    pred_mean.append(pred_mean[-1] + k * (y - pred_mean[-1]))
    pred_var.append((1-k) * pred_var[-1] + 1)

    reward = 1/ (pred_mean[-1] - y + 1e-6)

    

plt.plot(Y, label='reward')
plt.plot(X, label='ex_reward')
plt.plot(pred_mean, label='prediction')
plt.legend()
plt.show()

