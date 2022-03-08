"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in QLearning.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
import tkinter as tk
from QLearning import QLearningTable
from MDP import MDPTable


UNIT = 80   # pixels
BUTTON_H = 50  # button height
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width
HELL_STATES = [(0, 1), (1, 1), (1, 2), (0, 4), (1, 4), (3, 2), (3, 3), (5, 0)]
PARADISE_STATES = [(0, 2)]

def policy_evaluation():
    mdp_policy_evaluation()

def policy_update():
    mdp_policy_update()

def toggle_value_iteration():
    print("btn_toggle_value_iteration_clicked")
    mdp_value_iteration()

def reset():
    reset_Maze()

def run():
    run_QLearning()

def coords2xy(rect_coords):
    x = int((rect_coords[0] + rect_coords[2] - UNIT) / (UNIT * 2))
    y = int((rect_coords[1] + rect_coords[3] - UNIT - BUTTON_H * 2) / (UNIT * 2))
    return x, y

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT + BUTTON_H))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=MAZE_H * UNIT + BUTTON_H,
                                width=MAZE_W * UNIT)

        # create buttons
        self.btn_policy_evaluation = tk.Button(self.canvas,
                                               text="policy evaluation",
                                               command=policy_evaluation)
        self.btn_policy_evaluation.place(relx=0.01, rely=0.01, relwidth=0.24)

        self.btn_policy_update = tk.Button(self.canvas,
                                           text="policy update",
                                           command=policy_update)
        self.btn_policy_update.place(relx=0.25, rely=0.01, relwidth=0.24)

        self.btn_toggle_value_iteration = tk.Button(self.canvas,
                                                    text="value iteration",
                                                    command=toggle_value_iteration)
        self.btn_toggle_value_iteration.place(relx=0.50, rely=0.01, relwidth=0.24)
        # self.btn_reset = tk.Button(self.canvas,
        #                            text="reset",
        #                            command=reset)
        # self.btn_reset.place(relx=0.74, rely=0.01, relwidth=0.14)
        self.btn_run = tk.Button(self.canvas,
                                 text="Q-Learning",
                                 command=run)
        self.btn_run.place(relx=0.75, rely=0.01, relwidth=0.24)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0 + BUTTON_H, c, MAZE_H * UNIT + BUTTON_H
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r + BUTTON_H, MAZE_W * UNIT, r + BUTTON_H
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        self.origin = np.array([0.5 * UNIT, 0.5 * UNIT + BUTTON_H])

        # create q(s, a) labels
        self.q_labels = []
        for i in range(0, MAZE_H):
            labels_row = []
            for j in range(0, MAZE_W):
                center = self.origin + np.array([UNIT * i, UNIT * j])
                labels = [self.canvas.create_text(center[0], center[1] - 20,
                                                  text="",
                                                  width=30,
                                                  fill='green'),
                          self.canvas.create_text(center[0], center[1] + 20,
                                                  text="",
                                                  width=30,
                                                  fill='green'),
                          self.canvas.create_text(center[0] - 20, center[1],
                                                  text="",
                                                  width=30,
                                                  fill='green'),
                          self.canvas.create_text(center[0] + 20, center[1],
                                                  text="",
                                                  width=30,
                                                  fill='green')]
                labels_row.append(labels)
            self.q_labels.append(labels_row)

        # create v(s) labels
        self.v_labels = []
        for i in range(0, MAZE_H):
            labels_row = []
            for j in range(0, MAZE_W):
                center = self.origin + np.array([UNIT * i, UNIT * j])
                label = self.canvas.create_text(center[0], center[1] - 30,
                                                  text="",
                                                  fill='blue')
                labels_row.append(label)
            self.v_labels.append(labels_row)

        # create policy lines
        self.policy_arrows = []


        # create hells
        self.hell_centers = []
        for hell in HELL_STATES:
            self.hell_centers.append(self.origin + np.array([UNIT * hell[0], UNIT * hell[1]]))
        self.hells_coords = []
        for hell_center in self.hell_centers:
            hell = self.canvas.create_oval(
                hell_center[0] - 15, hell_center[1] - 15,
                hell_center[0] + 15, hell_center[1] + 15,
                outline='black')
            self.hells_coords.append(self.canvas.coords(hell))


        # create paradise
        self.paradise_centers = []
        for paradise in PARADISE_STATES:
            self.paradise_centers.append(self.origin + np.array([UNIT * paradise[0], UNIT * paradise[1]]))
        self.paradise_coords = []
        for paradise_center in self.paradise_centers:
            paradise = self.canvas.create_oval(
                paradise_center[0] - 15, paradise_center[1] - 15,
                paradise_center[0] + 15, paradise_center[1] + 15,
                outline='yellow')
            self.paradise_coords.append(self.canvas.coords(paradise))

        # create red rect
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            outline='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            outline='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT + BUTTON_H:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT + BUTTON_H:
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT


        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        # if s_ == self.canvas.coords(self.oval):
        if s_ in self.paradise_coords:
            reward = 10
            done = True
            s_ = 'Success'
        elif s_ in self.hells_coords:
            reward = -10
            done = True
            s_ = 'Fail'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()

    def fresh_labels(self):
        for row in self.q_labels:
            for col in row:
                for action in col:
                    self.canvas.dchars(action, 0, MAZE_W)
        for row in self.v_labels:
            for state in row:
                self.canvas.dchars(state, 0, MAZE_W)

def run_QLearning():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        steps = 0
        while True:
            # fresh env
            env.render()

            # QLearning choose action based on observation
            action = QLearning.choose_action(str(observation))

            # QLearning take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # QLearning learn from this transition
            QLearning.learn(str(observation), action, reward, str(observation_))

            # print Q-table
            x, y = coords2xy(list(observation))
            env.canvas.dchars(env.q_labels[x][y][action], 0, MAZE_W)
            env.canvas.insert(env.q_labels[x][y][action], 0, '%.2f' % QLearning.q_table.loc[str(observation), action])
            # env.q_labels[x][y][action].text = str(QLearning.q_table.loc[str(observation), action])

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print(("episode = %d, steps = %d, " + observation) % (episode, steps))
                break
            else:
                steps += 1

    # end of game
    print('game over')
    # env.destroy()


def reset_Maze():
    env.reset()
    env.fresh_labels()

def mdp_policy_evaluation():
    mdp.policy_evaluation()

    for i in range(MAZE_W):
        for j in range(MAZE_H):
            env.canvas.dchars(env.v_labels[i][j], 0, MAZE_W)
            env.canvas.insert(env.v_labels[i][j], 0, '%.2f' % mdp.value_table.loc["state-value", [(i, j)]])


def mdp_policy_update():
    mdp.policy_update()

    for arrow in env.policy_arrows:
        env.canvas.delete(arrow)
    for i in range(0, MAZE_H):
        for j in range(0, MAZE_W):
            center = env.origin + np.array([UNIT * i, UNIT * j])
            actions = mdp.policy_table[mdp.policy_table[(i, j)] > 0].index.to_list()
            for action in actions:
                env.policy_arrows.append(env.canvas.create_line(center[0], center[1],
                                                                center[0] + mdp.DIR[action][0] * 15, center[1] + mdp.DIR[action][1] * 15,
                                                                arrow=tk.LAST))


def mdp_value_iteration():
    mdp.value_iteration()

    for i in range(MAZE_W):
        for j in range(MAZE_H):
            env.canvas.dchars(env.v_labels[i][j], 0, MAZE_W)
            env.canvas.insert(env.v_labels[i][j], 0, '%.2f' % mdp.value_table.loc["state-value", [(i, j)]])

    for arrow in env.policy_arrows:
        env.canvas.delete(arrow)
    for i in range(0, MAZE_H):
        for j in range(0, MAZE_W):
            center = env.origin + np.array([UNIT * i, UNIT * j])
            actions = mdp.policy_table[mdp.policy_table[(i, j)] > 0].index.to_list()
            for action in actions:
                env.policy_arrows.append(env.canvas.create_line(center[0], center[1],
                                                                center[0] + mdp.DIR[action][0] * 15, center[1] + mdp.DIR[action][1] * 15,
                                                                arrow=tk.LAST))

if __name__ == "__main__":
    env = Maze()
    QLearning = QLearningTable(actions=list(range(env.n_actions)))
    mdp = MDPTable(actions=list(range(env.n_actions)),
                   bounds=(MAZE_W, MAZE_H),
                   hells=HELL_STATES,
                   paradises=PARADISE_STATES)
    env.mainloop()

