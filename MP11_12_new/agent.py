import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        self.N[state + (action,)] += 1

    def update_q(self, state, action, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        if state == None or action == None:
            return
        #self.update_n(state, action)
        N_value = self.N[state + (action,)]
        alpha = self.C / (self.C + N_value)
        Q_old = self.Q[state + (action,)]
        # print("state:", state, " action:", action)
        #print("Q old:", Q_old)
        Q_s_prime = [self.Q[s_prime + (act,)] for act in self.actions]
        
        max_Q_s_prime = max(Q_s_prime)

        self.Q[state + (action,)] += alpha * (r + self.gamma * max_Q_s_prime - Q_old)
        #print("after action ", action, ": ", Q_old + alpha * (r + self.gamma * max_Q_s_prime - Q_old))     

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        
        # TODO - MP12: write your function here
        
        if self._train and self.s != None and self.a != None:
            reward = -0.1
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1

            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)

        self.s = s_prime
        self.points = points

        if dead:
            self.reset()
            return 0
        
        Q_list = [self.Q[s_prime + (act,)] for act in self.actions]
        N_list = [self.N[s_prime + (act,)] for act in self.actions]

        Q_max_value = -100
        a_prime = utils.RIGHT
        for act in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            if self._train and N_list[act] < self.Ne:
                self.a = act
                return act
            elif Q_list[act] > Q_max_value:
                a_prime = act
                Q_max_value = Q_list[act]
        self.a = a_prime
        return a_prime

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        # state = {}
        food_dir_x, food_dir_y = self.check_food_dir(environment[0], environment[1], environment[3], environment[4])
        
        adjoining_wall_x, adjoining_wall_y = self.check_adjoining_wall(environment[0], environment[1], environment[5], environment[6])
        
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.check_adjoining_body(
            environment[0], environment[1], environment[2]
        )
        
        state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        # print("snake head:", (environment[0], environment[1]), "food:", (environment[3], environment[4]), "rock:", (environment[5], environment[6]))
        # print("state:", state)
        # print("===========================================")
        return state

    def check_adjoining_body(self, snake_head_x, snake_head_y, snake_body):
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        for x, y in snake_body:
            if snake_head_x == x + 1 and snake_head_y == y:
                adjoining_body_left = 1
            elif snake_head_x == x - 1 and snake_head_y == y:
                adjoining_body_right = 1

            if snake_head_y == y + 1 and snake_head_x == x:
                adjoining_body_top = 1
            elif snake_head_y == y - 1 and snake_head_x == x:
                adjoining_body_bottom = 1
        return adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right

    def check_adjoining_wall(self, snake_head_x, snake_head_y, rock_x, rock_y):
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        if snake_head_x == 1 or (snake_head_y == rock_y and snake_head_x == rock_x + 2): # wall or rock is on the left
            adjoining_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (snake_head_y == rock_y and snake_head_x == rock_x - 1): # wall or rock is on the right
            adjoining_wall_x = 2
        
        if snake_head_y == 1 or ((snake_head_x == rock_x or snake_head_x == rock_x + 1) and snake_head_y == rock_y + 1):
            adjoining_wall_y = 1
        elif snake_head_y == self.display_height - 2 or ((snake_head_x == rock_x or snake_head_x == rock_x + 1) and snake_head_y == rock_y - 1):
            adjoining_wall_y = 2
        
        return adjoining_wall_x, adjoining_wall_y

    def check_food_dir(self, snake_head_x, snake_head_y, food_x, food_y):
        food_dir_x = 0
        food_dir_y = 0
        if snake_head_x - food_x > 0: # food is on the left of head
            food_dir_x = 1
        elif snake_head_x - food_x < 0:
            food_dir_x = 2
        
        if snake_head_y - food_y > 0: # food is on the top of head
            food_dir_y = 1
        elif snake_head_y - food_y < 0:
            food_dir_y = 2
        return food_dir_x, food_dir_y
