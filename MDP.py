import numpy as np
import pandas as pd

eps = 0.0001

def get_states(bounds):
    states = []
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            states.append((x, y))
    return states

class MDPTable:
    def __init__(self, actions, bounds, hells, paradises, reward_decay=0.9):
        self.actions = actions
        self.bounds = bounds
        self.gamma = reward_decay
        self.states = get_states(bounds)
        self.DIR = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.paradises = paradises

        self.value_table = pd.DataFrame(columns=self.states, dtype=np.float64)
        self.init_state_value()
        self.init_reward(hells=hells, paradises=paradises)

        self.policy_table = pd.DataFrame(columns=self.states, dtype=np.float64)
        self.init_policy()

    def init_state_value(self):
        self.value_table = self.value_table.append(
            pd.Series(
                [0] * len(self.states),
                index=self.value_table.columns,
                name="state-value",
            )
        )

    def init_reward(self, hells, paradises):
        self.value_table = self.value_table.append(
            pd.Series(
                [0] * len(self.states),
                index=self.value_table.columns,
                name="reward",
            )
        )

        self.value_table.loc["reward", hells] += -1
        self.value_table.loc["reward", paradises] = 1

    def init_policy(self):
        for action in self.actions:
            self.policy_table = self.policy_table.append(
                pd.Series(
                    [1.0 / len(self.actions)] * len(self.states),
                    index=self.value_table.columns,
                    name=action,
                )
            )
        for action in self.actions:
            self.policy_table.loc[action, self.paradises] = 0

    def policy_evaluation(self):
        temp_value_table = pd.DataFrame.copy(self.value_table, deep=True)
        for state in self.states:
            sum = 0
            reward = float('%f' % temp_value_table.loc["reward", [state]])
            for action in self.actions:
                nstate = self.new_state(state, action)
                p = float('%f' % self.policy_table.loc[action, [state]])
                v = float('%f' % temp_value_table.loc["state-value", [nstate]])
                sum += p * v
            self.value_table.loc["state-value", [state]] = reward + self.gamma * sum

    def new_state(self, state, action):
        x = state[0] + self.DIR[action][0]
        y = state[1] + self.DIR[action][1]
        if min(x, y) < 0 or x >= self.bounds[0] or y >= self.bounds[1]:
            return state
        else:
            return x, y

    def policy_update(self):
        for state in self.states:
            if state in self.paradises:
                continue
            value_actions = []
            optimal_actions = []
            for action in self.actions:
                nstate = self.new_state(state, action)
                value_actions.append((float('%f' % self.value_table.loc["state-value", [nstate]]), action))
            max_value_action = max(value_actions)
            for value_action in value_actions:
                if value_action[0] == max_value_action[0]:
                    optimal_actions.append(value_action[1])
            for action in self.actions:
                if action in optimal_actions:
                    self.policy_table.loc[action, [state]] = 1.0 / len(optimal_actions)
                else:
                    self.policy_table.loc[action, [state]] = 0

    def value_iteration(self):
        max_iterations = 20
        for i in range(max_iterations):
            temp_value_table = pd.DataFrame.copy(self.value_table, deep=True)
            for state in self.states:
                q_sa = []
                reward = float('%f' % temp_value_table.loc["reward", [state]])
                for action in self.actions:
                    nstate = self.new_state(state, action)
                    if state in self.paradises:
                        p = 0
                    else:
                        p = 1
                    v = float('%f' % temp_value_table.loc["state-value", [nstate]])
                    q_sa.append(reward + self.gamma * p * v)
                self.value_table.loc["state-value", [state]] = max(q_sa)
            if np.sum(np.fabs(np.array(temp_value_table) - np.array(self.value_table))) <= eps:
                print("iteration rounds = ", i)
                break
        self.policy_update()
