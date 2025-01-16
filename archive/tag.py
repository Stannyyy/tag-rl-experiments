# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:11:00 2021

@author: StannyGoffin
"""

# Import packages
import rendering
import random
import numpy as np
from canvas import draw_state

# Game
class Tag:
    def __init__(self, GRID_SIZE, NUM_PLAYERS, NUM_TAGGERS):
        self._grid_size = GRID_SIZE
        self._num_play  = NUM_PLAYERS
        self._x_list    = [-1] * NUM_PLAYERS
        self._y_list    = [-1] * NUM_PLAYERS
        self._p_list    = list(range(NUM_PLAYERS))
        self._taggers   = [True] * NUM_TAGGERS + [False] * (NUM_PLAYERS - NUM_TAGGERS)
        self._options   = [0,1,2,3,4,5,6,7] # ['up','down','left','right','upleft','upright','downleft','downright']
        self._viewer    = None
        self._tot_turns = 0
        self.random_game()
    
    def random_game(self):
        self._x_list   = [-1] * self._num_play
        self._y_list   = [-1] * self._num_play

        for i in self._p_list:
            while True:
                x = int(np.floor(random.random()*(self._grid_size)))
                y = int(np.floor(random.random()*(self._grid_size)))
                check = [True for i in self._p_list if (self._x_list[i] == x) & (self._y_list[i] == y)]
                if len(check) == 0:
                    self._x_list[i] = x
                    self._y_list[i] = y
                    break
    
    # Move options        
    def what_options(self,turn):
        x = self._x_list[turn]
        y = self._y_list[turn]
        options = [0,1,2,3,4,5,6,7] # ['up','down','left','right','upleft','upright','downleft','downright']
            
        if y == 0:
            options[0] = -1
            options[4] = -1
            options[5] = -1
        if y == self._grid_size - 1:
            options[1] = -1
            options[6] = -1
            options[7] = -1
        if x == 0:
            options[2] = -1
            options[4] = -1
            options[6] = -1
        if x == self._grid_size - 1:
            options[3] = -1
            options[5] = -1
            options[7] = -1
            
        options = [o for o in options if o != -1]
        return options
    
    def random_move(self,turn,options):
        choice = random.sample(options,k=1)[0]
        reward = self.move(turn,choice)
        return choice, reward
    
    def move(self, turn, choice):
        x = self._x_list[turn]
        y = self._y_list[turn]
        if choice == 0: # ['up','down','left','right','upleft','upright','downleft','downright']
            y = y - 1
        elif choice == 1:
            y = y + 1
        elif choice == 2:
            x = x - 1
        elif choice == 3:
            x = x + 1
        elif choice == 4:
            y = y - 1
            x = x - 1
        elif choice == 5:
            y = y - 1
            x = x + 1
        elif choice == 6:
            y = y + 1
            x = x - 1
        elif choice == 7:
            y = y + 1
            x = x + 1
        
        if (x > (self._grid_size-1)) | (y > (self._grid_size-1)) | (x < 0) | (y < 0):
            print('Position out of bounds after move')
            print('X after move '+str(x))
            print('Y after move '+str(y))
            print('Choice: '+str(['up','down','left','right','upleft','upright','downleft','downright'][choice]))
            print('Options: '+str(self._options))
            raise Exception
            
        self._y_list[turn]   = y
        self._x_list[turn]   = x
        
        reward = self.what_reward(turn)
        return reward

    def what_reward(self,turn):
        is_tagger    = self._taggers[turn]
        x            = self._x_list[turn]
        y            = self._y_list[turn]
        in_same_spot = [i for i in range(self._num_play) if (self._x_list[i] == x) & (self._y_list[i] == y) & (i != turn)] 
        
        if is_tagger:
            reward = -0.5 * 2
        else:
            reward = 0.5
        
        for caught in in_same_spot:
            caught_is_tagger = self._taggers[caught]
            
            if is_tagger != caught_is_tagger:
                if is_tagger:
                    reward = 1.1 * self._grid_size
                else:
                    reward = -1.1 * self._grid_size * 2

        return reward
    
    # Render GUI
    def render(self,v_func=[1,1,1,1]):
        screen_size  = 400
        grid_size    = screen_size / (self._grid_size+2)
    
        if self._viewer is None:
            self._viewer = rendering.Viewer(screen_size, screen_size)
        
        # self._viewer.window.clear()
        self._viewer.geoms = []
                
        for l in range(len(self._x_list)):

            x = self._x_list[l]
            y = self._y_list[l]
            t = self._taggers[l]
                    
            if t == 1:
                team1 = rendering.make_circle(10)
                team1.add_attr(rendering.Transform(translation=(grid_size * (x + 1) + l * 3, screen_size - grid_size * (y + 1) + l * 3)))
                team1.set_color(1, 1, 0)
                self._viewer.add_geom(team1)
            elif t == 0:
                team0 = rendering.make_circle(10)
                team0.add_attr(rendering.Transform(translation=(grid_size * (x + 1) + l * 3, screen_size - grid_size * (y + 1) + l * 3)))
                team0.set_color(0, 0, 1)
                self._viewer.add_geom(team0)

        return self._viewer.render(return_rgb_array='human' == 'rgb_array')

    # Save state to image
    def save(self, debug_text, prefix = ''):
        draw_state(self, debug_text, prefix)
    
    # Shut down GUI
    def shut_down_GUI(self):
        self._viewer.close()