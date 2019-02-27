#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import tkinter as tk


class  Snakes_subsonic(object):

    def __init__(self):
        self.width = 10
        self.height = 5
        self.born_loc = np.array([5, 5])
        self.len = 3
        self.action = [0,1,2]
        self.count = 0
        self.MAXSTEP = 100
        self.try_times = []
        # 最多玩1000次
        # 0 什么也不做 1 相对于dir 左转 2相对于dir 右转

        self.first_render = True
        self.window_open = False

    def _ground_init(self, height, width):
        # 遇到 -1 game over
        # 0 表示空
        # 1表示snake身体
        # 2 表示食物
        self.ground = np.zeros([height + 2, width + 2])
        self.ground[0, :] = -1
        self.ground[-1, :] = -1
        self.ground[:, 0] = -1
        self.ground[:, -1] = -1
        # 食物位置 暂时一次性布置所有食物
        self.food_loc = np.array([[1, 5], [1, 1], [4, 8]])
        self._place_food(self.food_loc)

    def _snake_init(self, born_loc, length):
        #
        # dir 当前方向 0 上 1 下 2 左 3 右
        self.body = np.array([born_loc, born_loc - [0,1], born_loc - [0,2]])
        self.loc = self.body[0]
        for x,y in self.body:
            self.ground[x,y] = 1
        self.dir = 3

    def _place_food(self, food_loc):
        # 放置食物，用2来表示
        for x,y in food_loc:
            self.ground[x,y] = 2

    def step(self, action):
        done, reward = self._move(action)
        self.count += 1
        if done :
            self.try_times.append(self.count)

        return self.ground.flatten(), reward, done, {}


    def _move(self, action):
        # 方向朝上
        if self.dir == 0:
            loc_choice = np.array([self.loc + [-1, 0],
                                  self.loc + [0, -1],
                                  self.loc + [0, 1]])
        # 方向朝下
        elif self.dir == 1:
            loc_choice = np.array([self.loc + [1, 0],
                                  self.loc + [0, 1],
                                  self.loc + [0, -1]])
        # 方向朝左
        elif self.dir == 2:
            loc_choice = np.array([self.loc + [0, -1],
                                  self.loc + [1, 0],
                                  self.loc + [-1, 0]])
        # 方向朝右
        elif self.dir == 3:
            loc_choice = np.array([self.loc + [0, 1],
                                  self.loc + [-1, 0],
                                  self.loc + [1, 0]])
        else:
            raise NotImplementedError(" NO this dir, dir must be 0, 1, 2, 3!")
        
        self.loc_choice = loc_choice
        if action == 0:
            next_loc = loc_choice[0]
        elif action == 1:
            next_loc = loc_choice[1]
            if self.dir == 0:
                self.dir = 2
            elif self.dir == 1:
                self.dir = 3
            elif self.dir == 2:
                self.dir = 1
            elif self.dir == 3:
                self.dir = 0
        elif action == 2:
            next_loc = loc_choice[2]
            if self.dir == 0:
                self.dir = 3
            elif self.dir == 1:
                self.dir = 2
            elif self.dir == 2:
                self.dir = 0
            elif self.dir == 3:
                self.dir = 1

        # 检查next_loc 是什么
        #print(loc_choice)
        #print(next_loc)
        self.next_loc = next_loc
        content = self.ground[next_loc[0], next_loc[1]]

        done = False

        # 判断游戏是否达到最大step，以及是否完所有东西结束
        if (self.count >= self.MAXSTEP) or (not (self.ground == 2).sum()):
            done, reward = True, 0
        # 撞到墙或者吃到自己
        elif abs(content) == 1:
            done, reward = True, -10
        else:
            reward = self._follow_by_head(next_loc, content)

        self.loc = next_loc
        return done, reward

    def _follow_by_head(self, next_loc, content):
        # 没有吃到东西，往前走
        reward = 0
        if content == 0:
            self.ground[next_loc[0], next_loc[1]] = 1
            self.ground[self.body[-1][0], self.body[-1][1]] = 0
            self.body = np.append([next_loc], self.body[:-1], axis=0)
        elif content == 2:
            reward = 10
            self.ground[next_loc[0], next_loc[1]] = 1
            self.body = np.append([next_loc], self.body, axis=0)
        return reward

    def reset(self):
        self._ground_init(self.height, self.width)
        self._snake_init(self.born_loc, self.len)
        self.count = 0
        return self.ground.flatten()

    def render(self):
        if self.first_render:
            self.rg = Render_ground(self.width, self.height)
            self.window_open = True
            self.first_render = False
        self.rg.draw(self.body, self.ground)

    def close(self):
        if self.window_open:
            self.rg.window.destroy()
        return None

####
# render
class Render_ground():
    def __init__(self, width, height):
        self.border = 5.0
        self.scale = 10
        self.width = width * self.scale
        self.units = (self.width - 2 * self.border)/ width
        self.height = 2 * self.border + height * self.units

        self.window = tk.Tk()
        self.window.title('Snakes Subsonic')
        self.window.geometry('{0}x{1}'.format(int(self.width), int(self.height)))

    def _draw_ground(self, body, ground):
        self.can = tk.Canvas(self.window, width=self.width,
                              height=self.height)
        line_loc = [[self.border, self.border, self.width - self.border,self.border],
         [self.border, self.height - self.border, self.width - self.border, self.height - self.border],
         [self.border, self.border, self.border, self.height - self.border],
         [self.width - self.border, self.border, self.width - self.border, self.height - self.border]]

        self.line = [self.can.create_line(x0,y0,x1,y1) for x0,y0,x1,y1 in line_loc]
        # body
        body_loc = [[self.border + (y-1) * self.units, self.border +  (x-1) * self.units,
                     self.border + y * self.units, self.border +  x * self.units] for x,y in body]

        self.body = [self.can.create_rectangle(x0, y0, x1, y1,
                     fill='black') for x0,y0,x1,y1 in body_loc]
        # food
        idx = np.where(ground == 2)
        food = [[x,y] for x,y in zip(idx[0], idx[1])]
        food_loc = [[self.border + (y-1) * self.units, self.border +  (x-1) * self.units,
                     self.border + y * self.units, self.border +  x * self.units] for x,y in food]

        self.food = [self.can.create_rectangle(x0, y0, x1, y1,
                     fill='red') for x0,y0,x1,y1 in food_loc]

        self.can.pack()

    def draw(self, body, ground):
        self._draw_ground(body, ground)
        self.window.update()
        self.can.destroy()


