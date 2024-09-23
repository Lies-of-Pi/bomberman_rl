import os
import pickle
import random
import settings as s
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
import numpy as np
from scipy.special import softmax


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameter
n_nearest_coins = 3 #5 # number of nearest coins to be tracked
n_nearest_bombs = 1 # number of nearest bombs, maximum number is 4
epsilon = 0.5      # important for initialization of Q, can be a function of the iteration step?


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

    # if Q does not exist, create first empty matrix
    if not os.path.isfile("Q_saved.npy"):
        self.logger.info("Initialize a Q Matrix")

        # get the dimension of state space
        state_dimension = 16* 5**(n_nearest_coins + n_nearest_bombs) * 6**n_nearest_bombs

        self.Q = np.load("Q_saved.npy")


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
    state_index = feature_vector_to_state_index(features)
    
    # here we use a probabilistic model
    action_probabilities = softmax(self.Q[state_index, :] + epsilon * np.random.randn(len(ACTIONS))).tolist()

    return np.random.choice(ACTIONS, p=action_probabilities)


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

    # now we need the n nearest coins using the manhattan metric
    distances = []
    if len(game_state["coins"]) != 0:
        unsort_distances = np.array(manhattan_metric([(x_position, y_position)], game_state["coins"])[0])
        distances = np.sort(unsort_distances)

    # now we find the "direction" of the nearest coins:
    coin_directions = []
    for i in range(n_nearest_coins):
        if i >= len(distances): # coin does not exist
            # no direction exist
            coin_directions.append(-1)
        else:
            coin = game_state["coins"][np.where(unsort_distances == distances[i])[0][0]]
            # we look how many steps in each direction we have to go and so which direction is the most important
            coin_directions.append(np.argmax([np.maximum(0, x_position - coin[0]), np.maximum(0, coin[0] - x_position),
                                         np.maximum(0, y_position - coin[1]), np.maximum(0, coin[1] - y_position)]))

    # BOMB FEATURES ###############################################################################
    # now we need the n nearest bombs using the manhattan metric
    distances = []
    unsorted_distances = []
    for i in range(len(game_state["bombs"])):
        unsorted_distances.append(int(manhattan_metric([(x_position, y_position)], [game_state["bombs"][i][0]])[0][0]))
    unsorted_distances = np.array(unsorted_distances)
    distances = np.sort(unsorted_distances)

    # now we find the "direction" of the nearest bombs:
    bomb_directions = []
    for i in range(n_nearest_bombs):
        if i >= len(distances):  # bomb does not exist
            # no direction exist
            bomb_directions.append(-1)
        else:
            bomb = game_state["bombs"][np.where(unsorted_distances == distances[i])[0][0]][0]
            # we look how many steps in each direction we have to go and so which direction is the most important
            bomb_directions.append(np.argmax([np.maximum(0, x_position - bomb[0]), np.maximum(0, bomb[0] - x_position),
                                        np.maximum(0, y_position - bomb[1]), np.maximum(0, bomb[1] - y_position)]))

    # we want also the distances as features, where the value -1 means, that there is no bomb, 0 - 4 means the distance and a distance of 4+
    # is set to 4, because this is the safe distance
    bomb_distances = []
    for i in range(n_nearest_bombs):
        if i >= len(distances): # there is no bomb and no distance
            bomb_distances.append(-1)
        else:
            if distances[i] <= 4:
                bomb_distances.append(int(distances[i]))
            else:
                bomb_distances.append(4) # safe distance

    # ENVIRONMENT FEATURES ############################################################################

    # now we need a feature which shows where the next wall is
    field = game_state["field"]

    is_down_free = (field[y_position + 1][x_position] != -1)
    is_up_free = (field[y_position - 1][x_position] != -1)
    is_right_free = (field[y_position][x_position + 1] != -1)
    is_left_free = (field[y_position][x_position - 1] != -1)


    features = coin_directions + bomb_directions + [is_left_free, is_right_free, is_up_free, is_down_free] + bomb_distances # we merge four lists

    return features

def feature_vector_to_state_index(feature_vector: list):
    """
    if we use Q as matrix it is good to map our feature vectors to an index of the matrix
    here we use some kind of basis transformation
    """

    # map direction features to a number between 0 and 6**(n_nearest_coins + n_nearest_bombs) - 1
    index_direction = 0
    for i in range(n_nearest_coins + n_nearest_bombs):
        index_direction += (feature_vector[i] + 1) * 5**(n_nearest_coins + n_nearest_bombs - 1 - i)
    #print("direction: ", index_direction)

    # map environment features to a number between 0 and 15
    index_env = 0
    for i in range(4):
        index_env += feature_vector[i + n_nearest_coins + n_nearest_bombs] * 2**(3 - i)
    #print("env: ", index_env)

    # map bomb distances to a number between 0 and 7**(n_nearest_bombs) - 1
    index_bomb_distance = 0
    for i in range(n_nearest_bombs):
        index_bomb_distance += (feature_vector[i + n_nearest_coins + n_nearest_bombs + 4] + 1) * 6 ** (n_nearest_bombs - 1 - i)
    #print("bomb_dist ", index_bomb_distance)

    # map both indizes to a number
    # here we also use some kind of basis transformation but we have two elements with different ranges of values
    index1 = index_env + 16*index_direction # a number between 0 and 16 * (6**(n_coins + n_bombs)) - 1
    index = index1 + (6**(n_nearest_coins + n_nearest_bombs) - 1) * index_bomb_distance
    #index = index_env + 16 * index_coin
    return index

