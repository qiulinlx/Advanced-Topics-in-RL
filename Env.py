import gymnasium as gym
from gymnasium import spaces
import random
import pygame, sys
import render as render
from typing import List, Optional
import numpy as np 

def pos_to_integer(position, grid_size):
    row, col = position
    unique_integer = row * grid_size + col
    return unique_integer

class HuntingEnv(gym.Env):
    '''Custom Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=4):
        super(HuntingEnv, self).__init__()
        # Define action and observation space using gym.Spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.observation_space= spaces.Discrete(grid_size**2)
        
        # randomly initialise current state
        self.state=  [random.randint(0, 5) for _ in range(2)]
            
        # rewards 
        self.reward=0
        self.easy_target= [random.randint(0, 5) for _ in range(2)]
        self.hard_target= [random.randint(0, 5) for _ in range(2)]

        self.no_easy=0
        self.episode_length=0

    def get_action_meanings(self, action):
        action_list={0: "Move up", 1: "Move Down", 2:"Move Left", 3:"Move Right"}
        return action_list[action]
    
    def deer_step(self):
        """Define the movement of the easy and hard targets"""
        key=random.randint(0,4)
        if key <=2:
            direction_vector = [ self.state[0] -  self.easy_target[0],  self.state[1] -  self.easy_target[1]]  # Calculate the direction vector from the target to the agent
            magnitude = (direction_vector[0]**2 + direction_vector[1]**2)**0.5  # Calculate the magnitude (distance) of the direction vector
           
            if magnitude != 0:
                normalized_direction = [direction_vector[0] / magnitude, direction_vector[1] / magnitude]  # Normalize the direction vector (convert it to a unit vector)
            else:
                normalized_direction = [0, 0]   # Handle the case where the agent is already at the target's position

            self.easy_target= [round( self.easy_target[0] + normalized_direction[0]), round( self.easy_target[1] + normalized_direction[1])]
        else :
            self.easy_target=self.easy_target

        if key>3:
            direction_vector = [ self.state[0] -  self.hard_target[0],  self.state[1] -  self.hard_target[1]]  # Calculate the direction vector from the target to the agent
            magnitude = (direction_vector[0]**2 + direction_vector[1]**2)**0.5  # Calculate the magnitude (distance) of the direction vector
           
            if magnitude != 0:
                normalized_direction = [direction_vector[0] / magnitude, direction_vector[1] / magnitude]  # Normalize the direction vector (convert it to a unit vector)
            else:
                normalized_direction = [0, 0]   # Handle the case where the agent is already at the target's position

            self.hard_target= [round( self.hard_target[0] - normalized_direction[0]), round( self.hard_target[1] - normalized_direction[1])]
        if key==3:
             self.hard_target=self.hard_target
        if key ==4:
            action_key=random.randint(0,3)
            if action_key == 0: # Move up
               self.hard_target = (self.hard_target[0], self.hard_target[1]+1)
            elif action_key == 1: # Move down
               self.hard_target = (self.hard_target[0],self.hard_target[1]-1)
            elif action_key == 2: #Move left
                self.hard_target = (self.hard_target[0]-1, self.hard_target[1])
            elif action_key == 3: #Move right   
                self.hard_target = (self.hard_target[0]+1, self.hard_target[1])
        return self.easy_target, self.hard_target

    def step(self, action):
        '''defines the logic of your environment when the agent takes an actio
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
        info={}

        if action == 0: # Move up
            self.state = (self.state[0], self.state[1]+1)
        elif action == 1: # Move down
            self.state = (self.state[0], self.state[1]-1)
        elif action == 2: #Move left
            self.state = (self.state[0]-1, self.state[1])
        elif action == 3: #Move right   
            self.state = (self.state[0]+1, self.state[1])

        if self.state==self.easy_target:
            self.reward+=10
            self.no_easy+=1
            if self.no_easy<3:
                self.easy_target= (random.randint(1,6, size=2))
            if self.no_easy==3:
                self.easy_target= [10,10]

        if self.state==self.hard_target:
            self.reward+=20

        # define the completion of the episode
        if self.reward==60:
            done= True

            return self.state, self.reward, done, info
        # Generate unique integers for each cell

        obs=pos_to_integer(self.state, grid_size=4)
        return obs, self.reward, done, False, info #obs, reward, terminated, truncated, info 
  
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        #reset your environment
        self.reward=0
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos =[random.randint(0, 5) for _ in range(2)]
        
        self.agent_pos_obs=pos_to_integer(self.state, grid_size=4)

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(self.agent_pos_obs).astype(np.float32), {}  # empty info dict
  

    def render(self, mode='human', grid_size=6):
        # visualize the environment and agent intercation
        WINDOW_HEIGHT = grid_size*100
        WINDOW_WIDTH = grid_size*100
        render.main(self,grid_size)
        pygame.display.update()

    def close (self):
        pygame.quit()
        sys.exit()