# coding=utf-8

import pandas as pd
import numpy as np

"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import tk as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()

class Sarsa(object):
    def __init__(self, actions, env, alpha=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_if_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat(
                [self.q_table, pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                ).to_frame().T]
            )

    def get_next_action(self, state):
        self.check_if_state_exist(state)
        if np.random.rand() < self.epsilon:
            target_actions = self.q_table.loc[state, :]
            target_actions = target_actions.reindex(np.random.permutation(target_actions.index))
            target_action = target_actions.idxmax()
        else:
            target_action = np.random.choice(self.actions)
        return target_action

    def update_q_table(self, state, action, reward, state_next, action_next):
        self.check_if_state_exist(state_next)
        q_value_predict = self.q_table.loc[state, action]
        if state_next != 'terminal':
            q_value_real = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_value_real = reward
        self.q_table.loc[state, action] += self.alpha * (q_value_real - q_value_predict)

    def train(self):
        for episode in range(100):

            # Init state.
            state = self.env.reset()

            # Get first action.
            action = self.get_next_action(str(state))

            while True:
                self.env.render()

                # Get next state.
                state_next, reward, terminal = self.env.step(action)

                # Get next action.
                action_next = self.get_next_action(str(state_next))

                # Update Q table.
                self.update_q_table(str(state), action, reward, str(state_next), action_next)

                state, action = state_next, action_next

                if terminal:
                    break

            print('For episode: {}, the Q table is:\n {}'.format(episode, self.q_table))

        print('Game Over')
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    model = Sarsa(actions=list(range(env.n_actions)), env=env)
    env.after(50, model.train)
    env.mainloop()
