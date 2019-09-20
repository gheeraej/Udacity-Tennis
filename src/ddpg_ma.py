import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # Udpate every

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.ma = [Agent(state_size, action_size, i, random_seed) for i in range(n_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)
            
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i, agent in enumerate(self.ma):
                experiences = self.memory.sample()
                states, _, _, _, _ = experiences
                
                actions_target = [agent_bis.actor_target(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) for j, agent_bis in enumerate(self.ma)]
                actions_pred = [agent_bis.actor_local(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) for j, agent_bis in enumerate(self.ma)]
                    
                agent.learn(experiences, GAMMA, actions_target, actions_pred)
            
            for agent in self.ma:
                agent.soft_update(agent.critic_local, agent.critic_target, TAU)
                agent.soft_update(agent.actor_local, agent.actor_target, TAU)    
    
    
    def act(self, states, add_noise=True):
        actions = [np.squeeze(agent.act(np.expand_dims(state, axis=0), add_noise), axis=0) for agent, state in zip(self.ma, states)]
        return actions
       
        
    def reset(self):
        for agent in self.ma:
            agent.reset()
        
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
        
        
        
        
        
        
        
        
        