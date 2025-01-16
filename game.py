# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:11:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
from config import Config
from PIL import Image, ImageDraw, ImageFont
import copy
import glob
import os

# Game
class Game(Config):
    def __init__(self):

        # Import config
        Config.__init__(self)

        # Game variables
        self._x_list = [-1] * self.numPlayers
        self._y_list = [-1] * self.numPlayers
        self._taggers = [True] * self.numTaggers + [False] * (self.numPlayers - self.numTaggers)
        self._options = [0, 1, 2, 3,  # 0:up, 1:down, 2:left, 3:right,
                         4, 5, 6, 7]  # 4:up left, 5:up right, 6:down left, 7:down right
        self._ended = False

        # Initialize game
        self.init_random_game()

        # Initialize render
        self._rendered = ''
        self._prev_rendered = ''

        # Initialize save
        self.savePath = os.getcwd() + r'/results/'
    
    def init_random_game(self):
        self._x_list = [-1] * self.numPlayers
        self._y_list = [-1] * self.numPlayers

        for i in range(self.numPlayers):
            while True:
                x = int(np.floor(random.random()*self.gridSize))
                y = int(np.floor(random.random()*self.gridSize))
                check = [True for i in range(self.numPlayers) if (self._x_list[i] == x) & (self._y_list[i] == y)]
                if len(check) == 0:
                    self._x_list[i] = x
                    self._y_list[i] = y
                    break
        self._ended = False
    
    # Move options        
    def what_options(self, turn):

        # Get x and y position of the player whose turn it is
        x = self._x_list[turn]
        y = self._y_list[turn]

        # For different scenario's, rule out options
        options = copy.deepcopy(self._options)
        if y == 0:
            options[0] = -1
            options[4] = -1
            options[5] = -1
        if y == self.gridSize - 1:
            options[1] = -1
            options[6] = -1
            options[7] = -1
        if x == 0:
            options[2] = -1
            options[4] = -1
            options[6] = -1
        if x == self.gridSize - 1:
            options[3] = -1
            options[5] = -1
            options[7] = -1
            
        options = [o for o in options if o != -1]
        return options
    
    def move(self, turn, choice):    
        x = self._x_list[turn]
        y = self._y_list[turn]
        if choice == 0:
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
            
        self._y_list[turn] = y
        self._x_list[turn] = x
        
        reward = self.what_reward(turn)
        return reward

    def what_reward(self, turn):

        # Get state of player whose turn it is
        is_tagger = self._taggers[turn]
        x = self._x_list[turn]
        y = self._y_list[turn]

        # Check if player is in the same spot as another player
        in_same_spot = [i for i in range(self.numPlayers) if (self._x_list[i] == x) & (self._y_list[i] == y) & (i != turn)]

        # Usually, a tagger gets some punishment for each move, a runner gets some reward for each move
        if is_tagger:
            reward = -0.009 * self.gridSize
        else:
            reward = 0.009 * self.gridSize

        # When the tagger caught the runner, the tagger gets a large reward and the runner a large punishment
        for caught in in_same_spot:

            caught_is_tagger = self._taggers[caught]
            
            if is_tagger != caught_is_tagger:
                if is_tagger:
                    reward = 0.11 * self.gridSize
                else:
                    reward = -0.11 * self.gridSize

                self._ended = True

        return reward

    def render(self):

        playing_field = np.full(shape=(self.gridSize, self.gridSize), fill_value=' ')
        for tagger in np.where(self._taggers)[0].tolist():
            x_tagger = self._x_list[tagger]
            y_tagger = self._y_list[tagger]
            playing_field[x_tagger, y_tagger] = 'x'

        for runner in np.where([t == False for t in self._taggers])[0].tolist():
            x_runner = self._x_list[runner]
            y_runner = self._y_list[runner]
            if playing_field[x_runner, y_runner] == 'x':
                playing_field[x_runner, y_runner] = '%'
            else:
                playing_field[x_runner, y_runner] = 'o'

        # print(playing_field.T)
        self._rendered = playing_field.T
    
    def extract_position(self, grid, symbol):
        return [[x,y] for x in range(len(grid)) for y in range(len(grid[x])) if grid[x][y] == symbol]
    
    def draw_position(self, draw, players, prev_players, color, 
                      cell_size, player_radius, step_size):
        
        for i in range(len(players)):
            x = players[i][0]
            y = players[i][1]
            x_prev = prev_players[i][0]
            y_prev = prev_players[i][1]
            
            for j in range(1, step_size + 1):
                center_x = (x_prev + (x - x_prev) * j / step_size) * cell_size + cell_size // 2
                center_y = ((y_prev + (y - y_prev) * j / step_size) * cell_size) + 6 * 15 + cell_size // 2
                draw.ellipse(
                    [
                        center_x - player_radius,
                        center_y - player_radius,
                        center_x + player_radius,
                        center_y + player_radius,
                    ],
                    fill=color,
                )
                
        return draw

    def save(self, text, prefix):
        
        # Get player (prev) positions
        x_players = self.extract_position(self._rendered,'x') + self.extract_position(self._rendered,'%')
        o_players = self.extract_position(self._rendered,'o') + self.extract_position(self._rendered,'%')
        
        if str(self._prev_rendered) == '':
            self._prev_rendered = self._rendered
        
        x_prev_players = self.extract_position(self._prev_rendered,'x') + self.extract_position(self._prev_rendered,'%')
        o_prev_players = self.extract_position(self._prev_rendered,'o') + self.extract_position(self._prev_rendered,'%')
                
        # Set cell size and create an empty image
        cell_size  = 50
        grid_width = len(self._rendered)
        grid_height = len(self._rendered) + 3
        image_width = grid_width * cell_size
        image_height = grid_height * cell_size
        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)
        
        # Draw text
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
        text_margin = 5
        draw.text((text_margin, text_margin), text, fill=(0, 0, 0), font=font)

        # Draw grid lines
        for i in range(0, image_width, cell_size):
            draw.line([(i, 6 * 15), (i, image_height)], fill="black")
        for j in range(6 * 15, image_height, cell_size):
            draw.line([(0, j), (image_width, j)], fill="black")
        
        # Draw players
        player_radius = 20
        step_size = 5
        draw = self.draw_position(draw, x_players, x_prev_players, "red", 
                                   cell_size, player_radius, step_size)
        draw = self.draw_position(draw, o_players, o_prev_players, "blue", 
                                   cell_size, player_radius*0.8, step_size)

        # Save stationary image
        while len(prefix) < 3:
            prefix = '0'+prefix
            
        image.save(self.savePath+prefix+".png")
        
        # Save current state as previous
        self._prev_rendered = self._rendered

    def record(self, game_name):
        gif = []
        imgs = []
        for filename in glob.glob(self.savePath + '*.png'):
            pimg = Image.open(filename)
            imgs.append(pimg)
            imgs.append(pimg)
            imgs.append(pimg)

        for img in imgs:
            gif.append(img)

        gif[0].save(self.savePath + game_name + '.gif', save_all=True, optimize=False, append_images=gif[1:], loop=0)

        del gif
        del imgs
        del pimg
        del filename
        del img

        for filename in glob.glob(self.savePath + '*.png'):
            os.remove(filename)
