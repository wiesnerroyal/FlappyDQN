import numpy as np
import torch as T
from dqn import DQN
import random
from memory import Memory

class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size,
                    batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnr = replace
        self.learn_step_counter = 0

        self.memory = Memory(mem_size, n_actions)

        self.q_eval = DQN(self.lr)
        self.q_next = DQN(self.lr)

    def action(self, state):
        if np.random.random() > self.epsilon:
            actions = self.q_eval.forward(state)
            action_pure = T.argmax(actions).item()
            action = [1,0] if action_pure == 0 else [0,1]
        else:
            action = random.choice([[1,0],[0,1]])
            action_pure = 0 if action == [1,0] else 1
        
        return action, action_pure
    
    def save(self, state, action, reward, state_, done):
        self.memory.save(state, action, reward, state_, done)

    def load(self):
        state, action, reward, new_state, done = self.memory.load(self.batch_size)
        return state, action, reward, new_state, done

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnr == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.load()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()