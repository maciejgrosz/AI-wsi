import numpy as np
import gym
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


class q_learn:
    def __init__(self, env, lr, discount_factor, epsilon, ep, steps, policy, t=None):

        self.lr = lr
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.ep = ep
        self.steps = steps

        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.table = np.zeros((self.states, self.actions))
        self.policy = policy
        self.t = t

    def training(self):
        rews = []

        for episode in range(self.ep):
            state = self.env.reset()
            done = False
            rew = 0
            for _ in range(self.steps):
                if self.policy == 'eps':
                    if random.uniform(0, 1) < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.table[state, :])
                else:
                    action = self.boltzmann(state, self.t)

                # take action and get reward
                new_state, reward, done, info = self.env.step(action)

                # Q-learning algorithm
                self.table[state, action] = self.table[state, action] + self.lr * (
                        reward + self.discount_factor * np.max(self.table[new_state, :]) - self.table[state, action])

                # Update to our new state
                rew += reward
                state = new_state

                # if done, finish episode
                if done:
                    break
            rews.append(rew)
        return rews

    def results(self):
        state = self.env.reset()
        done = False
        rewards = 0

        for s in range(self.steps):
            print(f"TRAINED AGENT")
            print("Step {}".format(s + 1))
            # trained table
            action = np.argmax(self.table[state, :])
            new_state, reward, done, info = self.env.step(action)
            rewards += reward
            self.env.render()
            print(f"score: {rewards}")
            state = new_state
            if done:
                break

    def boltzmann(self, state, t):
        pt = [self.table[state, x] / t for x in range(self.actions)]
        prob_actions = np.exp(pt) / np.sum(np.exp(pt))
        cumulative_probability = 0.0
        choice = random.uniform(0, 1)
        for a, pr in enumerate(prob_actions):
            cumulative_probability += pr
            if cumulative_probability > choice:
                return a


def plot_rewards(num_episodes, rewards, policy):
    aver_rewards = []
    n = int(num_episodes * 0.1)
    for i in range(int(num_episodes / n)):
        aver_rewards.append(np.mean(rewards[n * i: n * (i + 1)]))
    x = range(1, num_episodes, n)
    y = aver_rewards
    # y = normalize([y])
    # y = sum(y)
    plt.plot(x, y, )
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.xscale('log')
    plt.title(policy + ' policy')

    plt.show()


def main():
    env = gym.make('Taxi-v3')
    learning_rate = 0.1
    discount_factor = 1
    epsilon = 0
    num_episodes = 10000
    max_steps = 1000
    policy = 'boltzmann'
    t = 1
    policy = 'eps'
    q = q_learn(env, learning_rate, discount_factor, epsilon, num_episodes, max_steps, policy, t)
    rewards = q.training()
    q.results()
    plot_rewards(num_episodes, rewards, policy)


if __name__ == '__main__':
    main()
