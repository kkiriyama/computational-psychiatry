import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm

class DrugOrTreat:
    def __init__(self, rewards, delays, dopamine):
        """set rewards and dopamin level"""
        self.rewards = rewards
        self.delays = delays
        self.dopamine = dopamine

    def step(self, state):
        """return an reward and completion flag against state"""
        done = False
        if state in [3, 4]:
            done = True

        reward = self.rewards[state]
        delay = self.delays[state]
        return reward, delay, done

    def __len__(self):
        """return number of states"""
        return len(self.rewards)

class AddictedActor:

    def __init__(self, env):
        """initialize actions"""
        self.actions = ['normal', 'drug']
        self.B = np.ones(6)
        self.num_actions = [0] * 2
        self.actor_log = [] 

    def selectAction(self, state):
        """return an selected action against a given state"""
        if state == 0:
            probs = self.B[1:3] / (np.sum(self.B[1:3], axis=0))
            try:
                a = np.random.choice(self.actions, 1, p=probs)
                return 1 if a[0] == 'normal' else 2
            except:
                return 1 if np.random.rand() > 0.5 else 2
        elif state == 1:
            return 3
        elif state == 2:
            return 4
        else:
            return 5

    def takeAction(self, action):
        """return whether the agent take selected action or not"""
        p = 1/(1+np.exp(1-self.B[action]))
        return np.random.choice([True, False], 1, p=[p, 1-p])

    def log(self, action):
        """log actions"""
        if action in [3, 4]:
            self.num_actions[action-3] += 1
            self.actor_log.append(copy.copy(self.num_actions))


class AddictedCritic():

    def __init__(self, env):
        """initialize state value"""
        self.V = np.zeros(len(env))
        self.critic_log = []

    def log(self):
        """log state value"""
        self.critic_log.append(copy.copy(self.V))


class ActorCritic():

    def __init__(self, actor_class, critic_class):
        """initialize actor and critic"""
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, max_episode=200, gamma=0.5, lr=0.05):
        """train and return logs"""
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        for e in tqdm(range(max_episode)):
            state = 0
            done = False
            while not done:
                selectedAction = actor.selectAction(state)
                takeAction = actor.takeAction(selectedAction)
                if takeAction:
                    next_state = selectedAction
                    reward, delay, done = env.step(state)
                    gain = reward + gamma ** delay * critic.V[next_state]
                    estimated = critic.V[state]

                    if state == len(env) - 2 and env.dopamine > 0:
                        td = max(gain - estimated + env.dopamine, env.dopamine)
                    else:
                        td = gain - estimated

                    actor.B[state] = 0.01 * actor.B[state] + 0.99 * td
                    critic.V[state] += lr * td
                    actor.log(state)
                    if state in [3, 4]:
                        critic.log()
                    state = next_state

        return actor.actor_log, critic.critic_log

def train_ac(
    max_episode=200,
    rewards=[0.0, 0.0, 0.0, 1.0, 0.8, 0.0],
    delays=[0, 3, 3, 1, 1, 20],
    gamma=0.5,
    dopamine=0.025
    ):
    """wrapper function of train"""
    trainer = ActorCritic(AddictedActor, AddictedCritic)
    env = DrugOrTreat(rewards=rewards, delays=delays, dopamine=dopamine)
    return trainer.train(env, max_episode=max_episode, gamma=gamma)

if __name__ == "__main__":
    NUM_ITER = 1000
    NUM_SIMULATIONS = 1
    alternative_ratios = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    action_ratio_light_addicted = []
    action_ratio_heavy_addicted= []

    for _ in range(NUM_SIMULATIONS):
        light_addicted = []
        heavy_addicted = []

        for ratio in alternative_ratios:
            actor_log, critic_log = train_ac(NUM_ITER, rewards=[0, 0, 0, ratio, 1, 0])
            light_addicted.append(sum([actor_log[i][1]/sum(actor_log[i]) for i in range(250, 500)])/250)
            heavy_addicted.append(sum([actor_log[i][1]/sum(actor_log[i]) for i in range(750, 1000)])/250)

        action_ratio_light_addicted.append(light_addicted)
        action_ratio_heavy_addicted.append(heavy_addicted)

    for i in range(NUM_SIMULATIONS):
        plt.scatter(alternative_ratios, action_ratio_light_addicted[i], color='g')
        plt.scatter(alternative_ratios, action_ratio_heavy_addicted[i], color='r')

    mean_light_addicted = np.mean(np.array(action_ratio_light_addicted), axis=0)
    mean_heavy_addicted = np.mean(np.array(action_ratio_heavy_addicted), axis=0)

    plt.plot(alternative_ratios, mean_light_addicted, label="step 250-500")
    plt.plot(alternative_ratios, mean_heavy_addicted, label="step 750-1000")
    plt.legend()
    plt.show()
