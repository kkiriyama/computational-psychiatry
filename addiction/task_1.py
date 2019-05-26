import matplotlib.pyplot as plt
import numpy as np
import copy

class DrugOrTreat:
    def __init__(self, rewards, dopamine):
        """set rewards and dopamin level"""
        self.rewards = rewards
        self.dopamine = dopamine

    def step(self, state):
        """return an reward and completion flag against state"""
        done = False
        if state in [3, 4]:
            done = True
        reward = self.rewards[state]
        return reward, done

    def __len__(self):
        """return number of states"""
        return len(self.rewards)

class AddictedActor():
    def __init__(self, env):
        """initialize actions"""
        self.actions = [1, 2]
        self.Q = np.zeros(2)
        self.num_actions = [0] * 2
        self.actor_log = []

    def policy(self, state):
        """return an action against a given state"""
        if state == 0:
            a = np.random.choice(self.actions, 1,
                                 p=np.exp(self.Q) / np.sum(np.exp(self.Q), axis=0))
            return a[0]
        elif state == 1:
            return 3
        elif state == 2:
            return 4
        else:
            return 5

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

    def train(self, env, max_episode=200, gamma=1, lr=0.1):
        """train and return logs"""
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        td_log = []

        for e in range(max_episode):
            state = 0
            done = False
            while not done:
                action = actor.policy(state)
                next_state = action
                reward, done = env.step(state)
                gain = reward + gamma * critic.V[next_state]
                estimated = critic.V[state]
                if state == len(env) - 2 and env.dopamine > 0:
                    td = max(gain - estimated + env.dopamine, env.dopamine)
                else:
                    td = gain - estimated
                td_log.append([e, state, td])
                if action in [1, 2]:
                    actor.Q[action - 1] += lr * td 
                critic.V[state] += lr * td
                if action in [3, 4]:
                    actor.log(action)
                if action == 5:
                    critic.log()
                state = next_state

        return actor.actor_log, critic.critic_log, td_log

def train_ac(max_episode=200, rewards=[0.0, 0.0, 0.0, 1.0, 0.8, 0.0], dopamine=0.1):
    """wrapper function of train"""
    trainer = ActorCritic(AddictedActor, AddictedCritic)
    env = DrugOrTreat(rewards=rewards, dopamine=dopamine)
    return trainer.train(env, max_episode=max_episode)

if __name__ == "__main__":
    NUM_ITER = 300
    actor_log, critic_log, td_log = train_ac(NUM_ITER)

    early = [[] for _ in range(5)]
    middle = [[] for _ in range(5)]
    late = [[] for _ in range(5)]

    for td in td_log:
        if td[0] in range(0, 100) and td[1] < 5:
            early[td[1]].append(td[2])
        if td[0] in range(100, 200) and td[1] < 5:
            middle[td[1]].append(td[2])
        if td[0] in range(200, 300) and td[1] < 5:
            late[td[1]].append(td[2])

    early = [sum(l)/len(l) for l in early]
    middle = [sum(l)/len(l) for l in middle]
    late = [sum(l)/len(l) for l in late]

    x = ['S0', 'S1', 'S2', 'S3', 'S4']
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.bar(x, early, color='g')
    ax1.set_xlabel("State")
    ax1.set_ylabel("TD Error in early stage(0-100)")
    ax1.set_ylim(0, 0.3)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(x, middle, color='g')
    ax2.set_xlabel("State")
    ax2.set_ylabel("TD Error in early stage(100-200)")
    ax2.set_ylim(0, 0.3)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(x, late, color='g')
    ax3.set_xlabel("State")
    ax3.set_ylabel("TD Error in late stage(200-300)")
    ax3.set_ylim(0, 0.3)
    plt.show()
