import os
import pickle
import random
import settings as s
from settings import COLS, ROWS
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
import numpy as np
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F
from .rule_based_agent import setup as rule_based_setup # to generate the first data
from .rule_based_agent import act as rule_based_act # to generate the first data
from .DeepQNetwork import DeepQNetwork


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# loading hyperparameters
with open('agent_code/QMan/hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # if there is no existing DeepQNetwork, then we create one
    if not os.path.isfile("DeepQNetwork.pth"):
        self.logger.info("Initialize a DeepQNetwork")

        self.Q_Network = DeepQNetwork()
        torch.save(self.Q_Network, "DeepQNetwork.pth")

    else:
        self.logger.info("Loading model from saved state.")
        self.Q_Network = torch.load("DeepQNetwork.pth")

    # initialize the target network
    self.target_network = torch.load("DeepQNetwork.pth")

    # check if we use the rule based agent
    if use_rule_based_agent:
        rule_based_setup(self)

    self.last_k_states = []


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # check if we use the rule based agent
    if use_rule_based_agent:
        return rule_based_act(self, game_state)

    else:
        # get the features
        features = state_to_features(self, game_state)

        # here we make a epsilon-greedy policy
        epsilon_greedy = np.random.rand()
        actual_epsilon = epsilon * epsilon_decay**(((game_state["round"] - 1) % epsilon_wavelength)/100)
        if epsilon_greedy <= actual_epsilon:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            q_values = self.Q_Network.forward(features).detach()
            probabilities = softmax(q_values - torch.max(q_values))
            return np.random.choice(ACTIONS, p=probabilities)
            #return ACTIONS[q_values.argmax()] # for deterministic model

def state_to_features(self, game_state: dict) -> np.array:
    """

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: torch.FloatTensor
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # our agents position on game field
    x_position, y_position = game_state["self"][3]

    # game field with values
    # 0: free, -1: wall, 1 crates,
    field = game_state["field"]

    # get all coins, value is 2
    for coin in game_state["coins"]:
        x_coin, y_coin = coin
        field[y_coin][x_coin] = 2

    # get all bombs, value is 3
    for bomb in game_state["bombs"]:
        x_bomb, y_bomb = bomb[0]
        field[y_bomb][x_bomb] = 3

    # get all explosions, value is 4
    explosion_mask = np.where(game_state["explosion_map"] == 1)
    field[explosion_mask] = 4

    # mark the figure, value is 5
    field[y_position][x_position] = 5

    # mark the other figures, value is 6
    for other in game_state["others"]:
        x_other, y_other = other[3]
        field[y_other][x_other] = 6

    # normalizazion
    field = (field + 1)/7 # all values are between 0 and 1

    # get the last k states
    if game_state["step"] == 1: # here we use the first situation as last frames
        for i in range(k_last_frames):
            self.last_k_states.append(field)

    feature = self.last_k_states[-k_last_frames:] + [field]

    # save the actual field as new last state
    self.last_k_states.append(field)

    return torch.FloatTensor(np.array(feature))

