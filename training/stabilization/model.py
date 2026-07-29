import torch
import torch.nn as nn
import random
import math

# GLOBAL PARAMETERS
BATCH_SIZE = 64
GAMMA = .99
EPS_START = .9
EPS_END = .01
EPS_DECAY = 2500
TAU = .005
LR = .00005


class QModel(nn.Module):
    def __init__(self, obs_dim, n_actions):
        """
        Create a Q-network for the DQN agent

        Args:
            obs_dim: number of observation dimensions
            n_actions: number of discrete actions
        """
        super().__init__()
        self.linear1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Compute Q-values from the input state

        Args:
            x: input state tensor

        Returns:
            Q-value tensor for each action
        """
        # Apply two hidden layers with ReLU activation
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    
# training: select action used in the loop
def select_action(state, policy_net, step_done, nb_actions):
    """
    Choose an action using epsilon-greedy policy

    Args:
        state: current state tensor
        policy_net: QModel used for Q-value estimation
        step_done: number of training steps completed
        nb_actions: total number of actions

    Returns:
        Selected action index
    """
    eps_threshold = EPS_END + \
        (EPS_START - EPS_END) * math.exp(-step_done / EPS_DECAY)

    # Exploration: randomly sample an action while epsilon is high
    if random.random() < eps_threshold:
        return random.randrange(nb_actions)

    # Exploitation: choose the action with highest Q-value
    with torch.no_grad():
        q_values = policy_net(state)
        return q_values.argmax().item()


def optimize_model(policy_net,
                   target_net,
                   optimizer,
                   memory):

    if len(memory) < BATCH_SIZE:
        return

    # Sample a batch of transitions for training
    states, actions, rewards, next_states, dones = \
        memory.sample(BATCH_SIZE)

    # Compute current Q-values for the taken actions
    current_q = policy_net(states).gather(1, actions)

    # Compute the target Q-values using the target network
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    criterion = nn.MSELoss()

    loss = criterion(current_q, target_q)




    optimizer.zero_grad()

    loss.backward()

    optimizer.step()