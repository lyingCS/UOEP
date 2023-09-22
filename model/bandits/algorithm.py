import numpy as np


class BetaBernoulliBandit:
    def __init__(self, arms=None, random_choices=5, verbose=False):
        self.arms = arms
        if arms is None:
            self.arms = [0, 0.5]
        self.arm = 1
        self.arm_value = self.arms[self.arm]
        self.random_choices = random_choices
        self.verbose = verbose

        self.alpha = [1 for _ in range(len(self.arms))]
        self.beta = [1 for _ in range(len(self.arms))]
        self.choices = [self.arm]
        self.rewards = [0]

    def sample(self):
        if len(self.choices) > self.random_choices:
            success_probs = np.random.beta(self.alpha, self.beta)
            self.arm = np.argmax(success_probs)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))
        self.arm_value = self.arms[self.arm]
        self.choices.append(self.arm)
        if self.verbose:
            print(f'Updated diversity importance coefficient, value: {self.arm_value}')
        return self.arm_value

    def update_dist(self, reward):
        prev_reward = self.rewards[-1]
        self.rewards.append(reward)
        reward = 1 if reward > prev_reward else 0
        self.alpha[self.arm] = self.alpha[self.arm] + reward
        self.beta[self.arm] = self.beta[self.arm] + 1 - reward

