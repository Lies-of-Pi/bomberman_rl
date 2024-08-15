<<<<<<< HEAD
import os
import pickle
import random
import settings as s
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
import numpy as np
from scipy.special import softmax
import torch
from torch import nn
from .rule_based_agent import setup as rule_based_setup # to generate the first data
from .rule_based_agent import act as rule_based_act # to generate the first data


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# loading hyperparameters
with open('hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters

# define our network architecture for the Deep-Q-Learning model
class DeepQNetwork(nn.Module):
    def __init__(self, n_input, n_hidden = 100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 6)
        )

    def forward(self, features):
        logits = self.linear_relu_stack(features)
        return logits


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

        self.Q_Network = DeepQNetwork(2*n_nearest_coins + (2*field_of_view_size + 1)**2 + 1)

        torch.save(self.Q_Network, "DeepQNetwork.pth")

    else:
        self.logger.info("Loading model from saved state.")
        self.Q_Network = torch.load("DeepQNetwork.pth")

    # check if we use the rule based agent
    if use_rule_based_agent:
        rule_based_setup(self)


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
        # todo Exploration vs exploitation
        features = state_to_features(game_state)

        # here we make a epsilon-greedy policy
        epsilon_greedy = np.random.rand()

        if epsilon_greedy <= epsilon * epsilon_decay**(game_state["round"]/100):
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            #print("feature: ", features)
            q_values = self.Q_Network.forward(features).detach()

            # Problem: obwohl sich die features ändern bleibt q_values konstant!!!!
            #print("Network:", q_values)
            probabilities = softmax(q_values - torch.max(q_values))
            #print(probabilities)
            return np.random.choice(ACTIONS, p=probabilities)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # our agents position on game field
    x_position, y_position = game_state["self"][3]


    # COIN FEATURES ################################################################################
    # for the coin features we use the difference in x and y to the coin position
    # this gives us the direction (in x and y) and the distance (0, 0) means that the coin does not exist
    # here we only look at the n nearest coins
    coin_features = []

    # 1. find the n nearest coins (using the manhattan metric)
    distances = []
    if len(game_state["coins"]) != 0: # if coins exist, we get the distances
        unsort_distances = np.array(manhattan_metric([(x_position, y_position)], game_state["coins"])[0])
        distances = np.sort(unsort_distances)[:n_nearest_coins]

    n_nearest_coins_indizes = []
    for coin_distance in distances:
        n_nearest_coins_indizes.append(np.where(unsort_distances == coin_distance)[0][0])

    # 2. get the directions for all existing nearest coins
    for index in n_nearest_coins_indizes:
        x_coin, y_coin = game_state["coins"][index]
        coin_features.append(x_coin - x_position)
        coin_features.append(y_coin - y_position)
    
    # 3. if not enough coins exist, we have to fill the feature vector with zeros:
    if (len(n_nearest_coins_indizes) < n_nearest_coins):
        for missing_coin in range(n_nearest_coins - len(n_nearest_coins_indizes)):
            coin_features.append(0)
            coin_features.append(0)
    #print(coin_features)


    # BOMB / ENVIRONMENT FEATURES ############################################################################
    # for the environment we use some kind of local field of view, so we look at a mxm playfield around our agent
    # with informations about the local environment
    # values:
    # -2: dangerous field
    # -1: stone wall
    # 0: free / safe place
    # 1: crates

    # 1. define all dangerous fields on the game field
    field = game_state["field"]
    field_x_size, field_y_size = field.shape

    # find the bomb zones
    for bomb_index in range(len(game_state["bombs"])):
        x_bomb, y_bomb = game_state["bombs"][bomb_index][0]

        # the explosion radius is 3 fields in each direction (look that all values are inside the field)
        x_dangerous = np.maximum(np.arange(x_bomb - 3, x_bomb + 4), np.ones(7))
        x_dangerous = np.minimum(x_dangerous, (field_x_size - 2) * np.ones_like(x_dangerous)).astype("int")
        y_dangerous = np.maximum(np.arange(y_bomb - 3, y_bomb + 4), np.ones(7)).astype("int")
        y_dangerous = np.minimum(y_dangerous, (field_y_size - 2) * np.ones_like(y_dangerous))


        field[y_dangerous, x_bomb] = -2
        field[y_bomb, x_dangerous] = -2

    # find the actual explosion_zones --> here no bomb is shown
    field[(game_state["explosion_map"] != 0).T] = -2

    # 2. to reduce feature space we only need a local field of view
    field_of_view = np.zeros((2*field_of_view_size + 1, 2*field_of_view_size + 1))

    x_indizes = np.maximum(np.arange(x_position - field_of_view_size, x_position + field_of_view_size + 1), np.zeros(2*field_of_view_size + 1))
    x_indizes = np.minimum(x_indizes, (field_x_size - 1) * np.ones_like(x_indizes)).astype("int")
    y_indizes = np.maximum(np.arange(y_position - field_of_view_size, y_position + field_of_view_size + 1), np.zeros(2*field_of_view_size + 1))
    y_indizes = np.minimum(y_indizes, (field_y_size - 1) * np.ones_like(y_indizes)).astype("int")

    # create field of view
    for i in range(2*field_of_view_size + 1):
        for j in range(2*field_of_view_size + 1):
            field_of_view[i][j] = field[y_indizes[i]][x_indizes[j]]

    # all features togetter
    features = np.concatenate((coin_features ,field_of_view.ravel(), [1])) # here we need the 1 to define the end state

    return torch.FloatTensor(features)

=======
import os
import pickle
import random
import settings as s
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
import numpy as np
from scipy.special import softmax
import torch
from torch import nn


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# loading hyperparameters
with open('hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters

# define our network architecture for the Deep-Q-Learning model
class DeepQNetwork(nn.Module):
    def __init__(self, n_input, n_hidden = 100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 6)
        )

    def forward(self, features):
        logits = self.linear_relu_stack(features)
        return logits


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

        self.Q_Network = DeepQNetwork(2*n_nearest_coins + (2*field_of_view_size + 1)**2 + 1)

        torch.save(self.Q_Network, "DeepQNetwork.pth")

    else:
        self.logger.info("Loading model from saved state.")
        self.Q_Network = torch.load("DeepQNetwork.pth")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # todo Exploration vs exploitation
    features = state_to_features(game_state)

    # here we make a epsilon-greedy policy
    epsilon_greedy = np.random.rand()

    if epsilon_greedy <= epsilon * epsilon_decay**game_state["step"]:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        q_values = self.Q_Network(features).detach()
        probabilities = softmax(q_values - torch.max(q_values))
        return np.random.choice(ACTIONS, p=probabilities)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # our agents position on game field
    x_position, y_position = game_state["self"][3]


    # COIN FEATURES ################################################################################
    # for the coin features we use the difference in x and y to the coin position
    # this gives us the direction (in x and y) and the distance (0, 0) means that the coin does not exist
    # here we only look at the n nearest coins
    coin_features = []

    # 1. find the n nearest coins (using the manhattan metric)
    distances = []
    if len(game_state["coins"]) != 0: # if coins exist, we get the distances
        unsort_distances = np.array(manhattan_metric([(x_position, y_position)], game_state["coins"])[0])
        distances = np.sort(unsort_distances)

    n_nearest_coins_indizes = []
    for coin_distance in distances:
        n_nearest_coins_indizes.append(np.where(unsort_distances == coin_distance)[0][0])

    # 2. get the directions for all existing nearest coins
    for index in n_nearest_coins_indizes:
        x_coin, y_coin = game_state["coins"][index]
        coin_features.append(x_coin - x_position)
        coin_features.append(y_coin - y_position)
    
    # 3. if not enough coins exist, we have to fill the feature vector with zeros:
    if (len(n_nearest_coins_indizes) < n_nearest_coins):
        for missing_coin in range(n_nearest_coins - len(n_nearest_coins_indizes)):
            coin_features.append(0)
            coin_features.append(0)
    #print(coin_features)


    # BOMB / ENVIRONMENT FEATURES ############################################################################
    # for the environment we use some kind of local field of view, so we look at a mxm playfield around our agent
    # with informations about the local environment
    # values:
    # -2: dangerous field
    # -1: stone wall
    # 0: free / safe place
    # 1: crates

    # 1. define all dangerous fields on the game field
    field = game_state["field"]
    field_x_size, field_y_size = field.shape

    # find the bomb zones
    for bomb_index in range(len(game_state["bombs"])):
        x_bomb, y_bomb = game_state["bombs"][bomb_index][0]

        # the explosion radius is 3 fields in each direction (look that all values are inside the field)
        x_dangerous = np.maximum(np.arange(x_bomb - 3, x_bomb + 4), np.ones(7))
        x_dangerous = np.minimum(x_dangerous, (field_x_size - 2) * np.ones_like(x_dangerous)).astype("int")
        y_dangerous = np.maximum(np.arange(y_bomb - 3, y_bomb + 4), np.ones(7)).astype("int")
        y_dangerous = np.minimum(y_dangerous, (field_y_size - 2) * np.ones_like(y_dangerous))


        field[y_dangerous, x_bomb] = -2
        field[y_bomb, x_dangerous] = -2

    # find the actual explosion_zones --> here no bomb is shown
    field[(game_state["explosion_map"] != 0).T] = -2

    # 2. to reduce feature space we only need a local field of view
    field_of_view = np.zeros((2*field_of_view_size + 1, 2*field_of_view_size + 1))

    x_indizes = np.maximum(np.arange(x_position - field_of_view_size, x_position + field_of_view_size + 1), np.zeros(2*field_of_view_size + 1))
    x_indizes = np.minimum(x_indizes, (field_x_size - 1) * np.ones_like(x_indizes)).astype("int")
    y_indizes = np.maximum(np.arange(y_position - field_of_view_size, y_position + field_of_view_size + 1), np.zeros(2*field_of_view_size + 1))
    y_indizes = np.minimum(y_indizes, (field_y_size - 1) * np.ones_like(y_indizes)).astype("int")

    # create field of view
    for i in range(2*field_of_view_size + 1):
        for j in range(2*field_of_view_size + 1):
            field_of_view[i][j] = field[y_indizes[i]][x_indizes[j]]

    # all features togetter
    features = np.concatenate((coin_features ,field_of_view.ravel(), [1])) # here we need the 1 to define the end state

    return torch.FloatTensor(features)

>>>>>>> 2763662b38ee7105bcdd017489e596d0297be8f2
