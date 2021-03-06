import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import anfis_codes.anfis
import copy

from anfis_codes.model import *
from rl.memory import *
from rl.prioritized_memory_replay import PrioritizedReplayBuffer


def averaging(model,input):
    #far
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf0'].c.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        model.layer['fuzzify'].varmfs[input].mfdefs['mf0'].c.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.copy_(torch.tensor(right,dtype=torch.float))

    #close far
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].b.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].b.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(right,dtype=torch.float))

    #near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].c.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        # if input == 'theta_lookahead':
        #     if left > -0.5:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].a.copy_(torch.tensor(-0.5,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].a.copy_(torch.tensor(left,dtype=torch.float))
        #     if right < 0.5:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].d.copy_(torch.tensor(0.5,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].d.copy_(torch.tensor(right,dtype=torch.float))
        # if input == 'theta_near':
        #     if left > -0.125:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].a.copy_(torch.tensor(-0.125,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].a.copy_(torch.tensor(left,dtype=torch.float))
        #     if right < 0.125:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].d.copy_(torch.tensor(0.125,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].d.copy_(torch.tensor(right,dtype=torch.float))
        # else:
        model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].c.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(right,dtype=torch.float))

    #close_near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        # if input == 'theta_lookahead':
        #     if left > -0.025:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(-0.025,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(left,dtype=torch.float))
        #     if right < 0.025:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(0.025,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(right,dtype=torch.float))
        # if input == 'theta_near':
        #     if left > -0.025:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(-0.025,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(left,dtype=torch.float))
        #     if right < 0.025:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(0.025,dtype=torch.float))
        #     else:
        #         model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(right,dtype=torch.float))
        # else:
        model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(right,dtype=torch.float))

def averaging7(model,input):
    #far
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf0'].c.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf6'].b.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        model.layer['fuzzify'].varmfs[input].mfdefs['mf0'].c.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf6'].b.copy_(torch.tensor(right,dtype=torch.float))

    #close far
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].b.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf5'].c.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].b.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf5'].c.copy_(torch.tensor(right,dtype=torch.float))

    #near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].c.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf5'].b.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        model.layer['fuzzify'].varmfs[input].mfdefs['mf1'].c.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf5'].b.copy_(torch.tensor(right,dtype=torch.float))

    #close_near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].c.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():

        model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].b.copy_(torch.tensor(left,dtype=torch.float))
        model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].c.copy_(torch.tensor(right,dtype=torch.float))

    #close_near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        if input == 'distance_line':
            const = 0.3
            if left > -const:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(-const,dtype=torch.float))
            else:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(left,dtype=torch.float))
            if right < const:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.copy_(torch.tensor(const,dtype=torch.float))
            else:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.copy_(torch.tensor(right,dtype=torch.float))
        else:
            model.layer['fuzzify'].varmfs[input].mfdefs['mf2'].c.copy_(torch.tensor(left,dtype=torch.float))
            model.layer['fuzzify'].varmfs[input].mfdefs['mf4'].b.copy_(torch.tensor(right,dtype=torch.float))

    #close_near
    left = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.item())
    right = abs(model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.item())
    avg = (left + right) / 2
    left = -avg
    right = avg
    with torch.no_grad():
        if input == 'distance_line':
            const = 0.001
            if left > -const:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(-const,dtype=torch.float))
            else:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(left,dtype=torch.float))
            if right < const:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(const,dtype=torch.float))
            else:
                model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(right,dtype=torch.float))
        # # elif input == 'theta_near':
        # #     if left > -0.025:
        # #         model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(-0.025,dtype=torch.float))
        # #     else:
        # #         model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(left,dtype=torch.float))
        # #     if right < 0.025:
        # #         model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(0.025,dtype=torch.float))
        # #     else:
        # #         model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(right,dtype=torch.float))
        else:
            model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].b.copy_(torch.tensor(left,dtype=torch.float))
            model.layer['fuzzify'].varmfs[input].mfdefs['mf3'].c.copy_(torch.tensor(right,dtype=torch.float))
def mfs_constraint(model):

    for i in range(len(model.input_keywords)):
        # for j in range(model.number_of_mfs[model.input_keywords[i]]):
        n_mfs = model.number_of_mfs[model.input_keywords[i]]
        if n_mfs == 5:
            averaging(model, model.input_keywords[i])

            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].b.item())

            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].c.item())

        if n_mfs == 7:
            averaging7(model, model.input_keywords[i])
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].c.item())

            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf2'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf3'].c.item())


            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf5'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf5'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf4'].c.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf5'].d.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf6'].b.item())
            model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf6'].a.data = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf5'].c.item())



        if n_mfs == 1: #it does not have a ZERO so 2-1
            left = abs(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d.item())
            right = abs(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b.item())
            avg = (left + right) / 2
            with torch.no_grad():
                model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d.copy_(torch.tensor(avg,dtype=torch.float))
                model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b.copy_(torch.tensor(avg,dtype=torch.float))

            # model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d.item())
            # model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf0'].d = torch.tensor(model.layer['fuzzify'].varmfs[model.input_keywords[i]].mfdefs['mf1'].b.item())


class DDPGagent:
    def __init__(self, num_inputs, num_outputs, anf, hidden_size=32, actor_learning_rate=1e-4,
    critic_learning_rate=1e-3, gamma=0.99, tau=1e-3, max_memory_size=50000, priority=True, grad_clip=1):
        # Params
        self.num_states = num_inputs
        #self.num_actions = env.action_space.shape
        self.num_actions = num_outputs
        self.gamma = gamma
        self.tau = tau
        self.curr_states = np.array([0.,0.,0.,0.,0.])
        # Networks
    #    self.actor = Actor(self.num_states, hidden_size, self.num_actions)
    #    self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor = anf
        self.actor_target = copy.deepcopy(anf)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.grad_clip = grad_clip
        # Training
        self.priority = priority
        if priority:
            self.memory = PrioritizedReplayBuffer(max_memory_size, .5)
        else:
            self.memory = Memory(max_memory_size)
        # self.critic_criterion  = torch.nn.MSELoss(reduction='sum')
        self.critic_criterion  = torch.nn.MSELoss()
        self.actor_optimizer  = optim.SGD(self.actor.parameters(), lr=actor_learning_rate, momentum=0.99)
    #    self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=critic_learning_rate, momentum=0.99)
    #    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action

    def update(self, batch_size):
        # states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        if self.priority:
            states, actions, rewards, next_states, _, weights, batch_idxes = self.memory.sample(batch_size, 0.5)
        else:
            states, actions, rewards, next_states, _ = self.memory.sample(batch_size, 0)
            weights, batch_idxes = np.ones_like(rewards), None

        weights = torch.FloatTensor(weights)

        states = torch.FloatTensor(states)
        #print(actions)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        actions = torch.reshape(actions,(batch_size,1))

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)/1.
        # critic_loss = self.critic_criterion(Qvals * weights, Qprime * weights)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()/-1.
        # policy_loss = self.critic.forward(states, self.actor.forward(states)).mean()
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        mfs_constraint(self.actor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        if self.priority:
            TD_error = torch.abs(Qprime - Qvals) + 1e-6

            self.memory.update_priorities(batch_idxes, TD_error)
