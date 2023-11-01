import gymnasium as gym
from gymnasium import spaces
import random
import pygame, sys
import render as render
from typing import List, Optional
import numpy as np 

class HuntingEnv(gym.Env):
    def __init__(self, size=6):

        super(HuntingEnv, self).__init__()

        self.size = size

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right, 4= pick axe up 5=kill deer
        self.action_space = spaces.Discrete(6)  

        # Observation space is grid of size:rows x columns
        # self.observation_space = ((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "axe position":  spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "easy target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "hard target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )
        # Initialize Pygame
        pygame.init()
        self.window_size = 512  # The size of the PyGame window

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
            return {"agent": np.array(self.current_pos), 
                    "easy target": np.array(self.easy_goal),
                    "hard target":  np.array(self.hard_goal),
                    "axe position": np.array(self.axe) }

    def reset(self,  seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.current_pos =  self.np_random.integers(0, self.size, size=2, dtype=int)
        
        self.easy_goal = self.current_pos
        while np.array_equal(self.easy_goal, self.current_pos):
            self.easy_goal = self.np_random.integers( 0, self.size, size=2, dtype=int)
        
        self.hard_goal = self.current_pos
        while np.array_equal(self.hard_goal, self.current_pos) or np.array_equal(self.hard_goal, self.easy_goal):
                self.hard_goal = self.np_random.integers( 0, self.size, size=2, dtype=int)
        
        self.axe = self.hard_goal
        while np.array_equal(self.axe, self.hard_goal) or np.array_equal(self.axe, self.easy_goal) or np.array_equal(self.axe, self.current_pos):
                self.axe = self.np_random.integers( 0, self.size, size=2, dtype=int)

        self.holding_axe=False
        
        self.reward= 0
        observation = self._get_obs()
        info= dict()

        return observation, info

    def step(self, action):

        action_to_direction = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1]),
                4: np.array([0, 0]),
                5: np.array([0, 0])
                }

        direction = action_to_direction[int(action)]

        # We use `np.clip` to make sure we don't leave the grid
        self.current_pos = np.clip(self.current_pos + direction, 0, self.size - 1)
         # Map the action (element of {0,1,2,3}) to the direction we walk in

        if action == 4:
            if np.array_equal(self.axe, self.current_pos):
                self.reward= self.reward + 50
                self.holding_axe=True
                print("Axe obtained")
            else: 
                self.reward= self.reward - 10
                print("Axe not found")

        if action == 5:
            if self.holding_axe==True:
                if np.array_equal(self.easy_goal, self.current_pos) or np.array_equal(self.hard_goal, self.current_pos): 
                    self.reward= self.reward + 20
                    print("one deer killed")
            else: 
                self.reward= self.reward - 10   
                print("Hunt failed")

        terminated = np.array_equal(self.hard_goal, self.current_pos) or np.array_equal(self.easy_goal, self.current_pos) and action == 5

        #Prey moves randomly in the grid
        easy, hard=self.deer_step(self.current_pos, self.easy_goal, self.hard_goal)
        self.easy_goal=easy
        self.hard_goal=hard
        
        self.easy_goal = np.clip(self.easy_goal + direction, 0, self.size - 1)
        self.hard_goal = np.clip(self.hard_goal + direction, 0, self.size - 1)

        observation = self._get_obs()
        info = dict()
        self.reward= self.reward - 1
        return observation, self.reward, terminated, False, info
    
    def deer_step(self, current, easy, hard):
        
        key=random.randint(0,10)
        if key ==0:
            direction_vector = [ current[0] -  easy[0],  current[1] -  easy[1]]  # Calculate the direction vector from the target to the agent
            magnitude = (direction_vector[0]**2 + direction_vector[1]**2)**0.5  # Calculate the magnitude (distance) of the direction vector
           
            if magnitude != 0:
                normalized_direction = [direction_vector[0] / magnitude, direction_vector[1] / magnitude]  # Normalize the direction vector (convert it to a unit vector)
            else:
                normalized_direction = [0, 0]   # Handle the case where the agent is already at the target's position

            easy= [round( easy[0] + normalized_direction[0]), round( easy[1] + normalized_direction[1])]
        else :
            easy=easy

        if key>3:
            direction_vector = [ current[0] -  hard[0],  current[1] -  hard[1]]  # Calculate the direction vector from the target to the agent
            magnitude = (direction_vector[0]**2 + direction_vector[1]**2)**0.5  # Calculate the magnitude (distance) of the direction vector
           
            if magnitude != 0:
                normalized_direction = [direction_vector[0] / magnitude, direction_vector[1] / magnitude]  # Normalize the direction vector (convert it to a unit vector)
            else:
                normalized_direction = [0, 0]   # Handle the case where the agent is already at the target's position

            hard= [round( hard[0] - normalized_direction[0]), round( hard[1] - normalized_direction[1])]
        if key==3:
            hard=hard
        if key <4:
            action_key=random.randint(0,3)
            if action_key == 0: # Move up
               hard = (hard[0], hard[1]+1)
            elif action_key == 1: # Move down
               hard = (hard[0],hard[1]-1)
            elif action_key == 2: #Move left
                hard = (hard[0]-1, hard[1])
            elif action_key == 3: #Move right   
                hard = (hard[0]+1, hard[1])
        return easy, hard
      
    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True

    def render(self, mode='human', grid_size=6):
        WINDOW_HEIGHT = grid_size*100
        WINDOW_WIDTH = grid_size*100
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        GREEN =( 60,179,113)
        cell_size = 100
        window_size = (grid_size * cell_size, grid_size * cell_size)
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((window_size))
        self.clock = pygame.time.Clock()

        canvas = pygame.Surface(( WINDOW_HEIGHT, WINDOW_WIDTH))
        canvas.fill((GREEN))
        blockSize = int(WINDOW_HEIGHT/grid_size) #Set the size of the grid block
        for x in range(0, grid_size):
            for y in range(0, grid_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(canvas, WHITE, rect, 1)
        
        new_width = WINDOW_WIDTH/grid_size
        new_height = WINDOW_WIDTH/grid_size

        hunter = pygame.image.load('hunter.png')
        block_x = self.current_pos[1] * blockSize
        block_y = self.current_pos[0] * blockSize
        hunter, (image_x, image_y) =render.centering(hunter, new_width, new_height, blockSize, block_x, block_y)
        canvas.blit( hunter, (image_x, image_y))

        axe = pygame.image.load('axe.png')
        block_x = self.axe[1] * blockSize
        block_y =self.axe[0] * blockSize
        axe, (image_x, image_y) =render.centering(axe, new_width, new_height, blockSize, block_x, block_y)
        canvas.blit( axe, (image_x, image_y))

        deer = pygame.image.load('deer1.png')
        block_x = self.hard_goal[1] * blockSize
        block_y = self.hard_goal[0] * blockSize
        deer, (image_x, image_y) =render.centering(deer, new_width, new_height, blockSize, block_x, block_y)
        canvas.blit( deer, (image_x, image_y))


        bunny = pygame.image.load('bunny.png')
        block_x = self.easy_goal[1] * blockSize
        block_y =self.easy_goal[0] * blockSize
        bunny, (image_x, image_y) =render.centering(bunny, new_width, new_height, blockSize, block_x, block_y)
        canvas.blit( bunny, (image_x, image_y))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(2)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()