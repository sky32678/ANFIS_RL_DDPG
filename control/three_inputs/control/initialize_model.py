import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import anfis_codes.anfis
from rl.ddpg import DDPGagent
from rl.memory import *
from anfis_codes.model import *

anf = Anfis().my_model()
#print(env.action_space.shape)
#env = gym.make('CartPole-v1')
num_inputs, num_outputs = 3, 1
agent = DDPGagent(num_inputs, num_outputs, anf, 32, 1e-5*5, 1e-3)
actor = agent.actor
critic = agent.critic
torch.save(agent, 'anfis_initialized.model')
#torch.save(agent, 'actor.npy',_use_new_zipfile_serialization=False)
#torch.save(agent, 'critic.npy',_use_new_zipfile_serialization=False)
print(agent.curr_states)
# print(agent.critic.linear1.weight)
dd
#noise = OUNoise(env.action_space)
