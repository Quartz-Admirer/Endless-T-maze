import numpy as np
import random

class EndlessTMaze:
    def __init__(self, corridor_length=10, num_corridors=3, penalty=0, seed=None, goal_reward=1):
        self.corridor_length = corridor_length
        self.num_corridors = num_corridors
        self.penalty = penalty
        self.goal_reward = goal_reward
        self.max_steps = (corridor_length + 1) * num_corridors
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self):
        self.current_corridor = 0
        self.x = 0
        self.y = 0
        self.done = False
        self.steps = 0
        self.total_reward = 0
        
        self.hints = [random.choice([0, 1]) for _ in range(self.num_corridors)]
        self.current_hint = self.hints[0]
        
        return self.get_state()
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        
        self.steps += 1
        reward = 0
        
        if self.x < self.corridor_length:
            if action == 3: 
                self.x += 1
            else:
                reward = self.penalty
                self.done = True
        
        elif self.x == self.corridor_length:
            if action == 0:
                self.y = 1
            elif action == 1:
                self.y = -1
            else:
                reward = self.penalty
                self.done = True
            
            if not self.done:
                correct_action = self.current_hint
                if action == correct_action:
                    reward = self.goal_reward
                    self.total_reward += reward
                    self.current_corridor += 1
                    
                    if self.current_corridor < self.num_corridors:
                        self.x = 0
                        self.y = 0
                        self.current_hint = self.hints[self.current_corridor]
                    else:
                        self.done = True
                else:
                    reward = self.penalty
                    self.done = True
        
        if not self.done and self.steps >= self.max_steps:
            self.done = True
        
        return self.get_state(), reward, self.done, {"total_reward": self.total_reward}
    
    def get_state(self):
        hint_to_show = self.current_hint if self.x == 0 else 0
        return np.array([self.x, self.y, hint_to_show], dtype=np.float32)

    def get_optimal_action(self, state, in_corridor):
        if self.x < self.corridor_length:
            return 3, None
        else:
            return self.current_hint, None