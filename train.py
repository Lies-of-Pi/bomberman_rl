<<<<<<< HEAD
from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
# python main.py play --no-gui --train 1 --n-rounds 3000 --agents coin_hunter_ag
# ent --scenario coin-heaven

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS_DICTIONARY = dict(zip(ACTIONS, [0, 1, 2, 3, 4, 5]))

# loading hyperparameters
with open('hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # define optimizer, here we use Stochastic Gradient Descent
    self.optimizer = torch.optim.SGD(self.Q_Network.parameters(), lr=learning_rate)
    self.writer = SummaryWriter()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # get the features to describe the state
    s = state_to_features(old_game_state)
    s_new = state_to_features(new_game_state)

    # get the reward
    r = reward_from_events(self, events)

    # define additional rewards
    # if we stand in a dangerous field, this is bad
    dangerous_field = (s_new[2*n_nearest_coins + field_of_view_size] == -2)
    if dangerous_field:
        r -= 0.3
    else:
        r += 0.1

    # if we get closer to a coin, this is good
    old_coin_distances = s[0: 2*n_nearest_coins].reshape(-1, 2).numpy()
    old_coin_distances = manhattan_metric(old_coin_distances, np.zeros_like(old_coin_distances))[:, 0]

    new_coin_distances = s_new[0: 2 * n_nearest_coins].reshape(-1, 2).numpy()
    new_coin_distances = manhattan_metric(new_coin_distances, np.zeros_like(new_coin_distances))[:, 0]

    difference = old_coin_distances - new_coin_distances # positiv, if we get closer to the coin
    r += np.sum(difference * 0.1) # here a negative difference reduces the reward

    # restore the states, actions and rewards
    self.transitions.append(Transition(s, self_action, s_new, r))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    # preparation of the trainings data ######################################################################################

    # 1. check for additional training data
    if os.path.isfile("training_data.npz"):
        self.logger.info("loading additional training data")
        loading = np.load("training_data.npz")
        s_load = loading["s"]
        a_load = loading["a"]
        s_new_load = loading["s_new"]
        r_load = loading["r"]
        r_cumsum_load = loading["r_cumsum"]
        s_new_plus_k_load = loading["s_new_plus_k"]
    else:
        s_load = []
        a_load = []
        s_new_load = []
        r_load = []
        r_cumsum_load = []
        s_new_plus_k_load = []

    # 2. load the data from the last game
    total_reward = 0
    s_list = []
    a_list = []
    s_new_list = []
    r_list = []

    for tuple in self.transitions:
        s, a, s_new, r = tuple
        total_reward += r

        # store all features
        s_list.append(s)
        a_list.append(ACTIONS_DICTIONARY[a])
        r_list.append(r)
        
        # important s_new = None is a special case, here we define an other vector
        if s_new == None:
            # define endstate value, just for data storage
            s_new = torch.zeros(2*n_nearest_coins + (2*field_of_view_size + 1)**2 + 1)
        s_new_list.append(s_new)

    # updating the Network #######################################################################################
    # 1. using the data from the last game
    r_cumsum_list = []
    s_new_plus_k_list = []
    for i in range(len(s_list)):
        s = s_list[i]
        a = a_list[i]
        s_new = s_new_list[i]

        # get cumulated reward
        r_cumsum = np.array(r_list[i: np.min([i + k, len(r_list)])])
        gamma_factor = gamma ** np.arange(len(r_cumsum))
        r_cumsum = np.sum(gamma_factor * r_cumsum)

        s_new_plus_k = s_new_list[np.min([i + k, len(s_new_list) - 1])]

        r_cumsum_list.append(r_cumsum)
        s_new_plus_k_list.append(s_new_plus_k)

        # update Q Function
        self.logger.info("update Q Function with last game")
        if s_new_plus_k[-1] == 0: # end state
            #print(r_cumsum)
            loss = (r_cumsum - self.Q_Network.forward(s)[a])**2
        else:
            loss = (r_cumsum + gamma**k * torch.max(self.Q_Network.forward(s_new_plus_k)) - self.Q_Network.forward(s)[a])**2
            #print(loss)

        # Backpropagation
        loss.backward()

        #prevent exploding gradients
        nn.utils.clip_grad_norm_(self.Q_Network.parameters(), max_gradient_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # restore the loss
        self.writer.add_scalar("Loss/train", loss, last_game_state["round"])


    # 2. using the data from the old games
    if (len(s_load) >= additional_training_set_size) and (len(s_load) != 0):
        batch_indizes = np.random.permutation(len(s_load))[:additional_training_set_size]

    else:
        batch_indizes = np.arange(len(s_load))

    if len(s_load) != 0:
        for i in batch_indizes:
            s = torch.FloatTensor(s_load[i])
            a = a_load[i]
            s_new = torch.FloatTensor(s_new_load[i])
            r = r_load[i]
            r_cumsum = r_cumsum_load[i]
            s_new_plus_k = torch.FloatTensor(s_new_plus_k_load[i])

            # update Q Function
            self.logger.info("update Q Function with old games")
            if s_new_plus_k[-1] == 0:
                loss = (r_cumsum - self.Q_Network.forward(s)[int(a)]) ** 2
            else:
                loss = (r_cumsum + gamma ** k * torch.max(self.Q_Network.forward(s_new_plus_k)) - self.Q_Network.forward(s)[int(a)]) ** 2

            # Backpropagation
            loss.backward()

            # prevent exploding gradients
            nn.utils.clip_grad_norm_(self.Q_Network.parameters(), max_gradient_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            # restore the loss
            self.writer.add_scalar("Loss/train", loss, last_game_state["round"])

    # Store the model
    self.logger.info("save the Q Network")
    torch.save(self.Q_Network, "DeepQNetwork.pth")
    self.writer.flush()
    self.writer.close()

    model_scripted = torch.jit.script(self.Q_Network)  # Export to TorchScript
    model_scripted.save('model_scripted.pt')  # Save


    # export the training data ########################################################################################
    self.logger.info("export training data")

    # concatenate loaded training data and new data
    s_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_load), np.ravel(s_list))).reshape(-1, len(s_list[0])))
    a_all_data = np.concatenate((a_load, a_list))
    s_new_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_new_load), np.ravel(s_new_list))).reshape(-1, len(s_list[0])))
    r_all_data = np.concatenate((r_load, r_list))
    r_cumsum_all_data = np.concatenate((r_cumsum_load, r_cumsum_list))
    s_new_plus_k_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_new_plus_k_load),
                                                              np.ravel(s_new_plus_k_list))).reshape(-1, len(s_list[0])))


    # export data
    np.savez("training_data.npz", s=s_all_data, a=a_all_data, s_new=s_new_all_data, r=r_all_data,
             r_cumsum=r_cumsum_all_data, s_new_plus_k=s_new_plus_k_all_data)


    # export quality data #############################################################################################
    dead = ("KILLED_SELF" in events)
    end_score = last_game_state["self"][1]
    number_of_steps = last_game_state["step"]

    # check if quality data exists:
    if os.path.isfile("training_qualities.npz"):
        loading = np.load("training_qualities.npz")
        dead_load = loading["dead"]
        end_score_load = loading["end_score"]
        number_of_steps_load = loading["number_of_steps"]
        total_reward_load = loading["total_reward"]
    else:
        dead_load = []
        end_score_load = []
        number_of_steps_load = []
        total_reward_load = []

    # restore quality data
    all_dead_data = np.concatenate((dead_load, [dead]))
    all_end_score_data = np.concatenate((end_score_load, [end_score]))
    all_number_of_steps_data = np.concatenate((number_of_steps_load, [number_of_steps]))
    all_total_reward_data = np.concatenate((total_reward_load, [total_reward]))

    np.savez("training_qualities.npz", dead=all_dead_data, end_score=all_end_score_data, number_of_steps=all_number_of_steps_data,
             total_reward=all_total_reward_data)

    # reset transition store for the next game
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # good events
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 10,

        # really bad events
        e.INVALID_ACTION: -0.5, # so we learn to navigate and not walking to walls
        e.KILLED_SELF: -10, # this is really bad

        # bad in this context
        e.WAITED: -0.1  # for this task we want to move with minimal number of steps

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
=======
from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from sklearn.metrics.pairwise import manhattan_distances as manhattan_metric
#from .callbacks import n_nearest_bombs, n_nearest_coins
import os
import torch
# python main.py play --no-gui --train 1 --n-rounds 3000 --agents coin_hunter_ag
# ent --scenario coin-heaven

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS_DICTIONARY = dict(zip(ACTIONS, [0, 1, 2, 3, 4, 5]))

# loading hyperparameters
with open('hyperparameters.txt') as f:
    lines = f.readlines()
for hyperparameter in lines:
    exec(hyperparameter[:-1]) # so we get all hyperparameters



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.transitions_state_indizes = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # define optimizer, here we use Stochastic Gradient Descent
    self.optimizer = torch.optim.SGD(self.Q_Network.parameters(), lr=learning_rate)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # get the features to describe the state
    s = state_to_features(old_game_state)
    s_new = state_to_features(new_game_state)

    # get the reward
    r = reward_from_events(self, events)

    # define additional rewards
    # if we stand in a dangerous field, this is bad
    dangerous_field = (s_new[2*n_nearest_coins + field_of_view_size] == -2)
    if dangerous_field:
        r -= 0.3
    else:
        r += 0.1

    # if we get closer to a coin, this is good
    old_coin_distances = s[0: 2*n_nearest_coins].reshape(-1, 2).numpy()
    old_coin_distances = manhattan_metric(old_coin_distances, np.zeros_like(old_coin_distances))[:, 0]

    new_coin_distances = s_new[0: 2 * n_nearest_coins].reshape(-1, 2).numpy()
    new_coin_distances = manhattan_metric(new_coin_distances, np.zeros_like(new_coin_distances))[:, 0]

    difference = old_coin_distances - new_coin_distances # positiv, if we get closer to the coin
    r += np.sum(difference * 0.2) # here a negative difference reduces the reward

    # restore the states, actions and rewards
    self.transitions.append(Transition(s, self_action, s_new, r))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    # preparation of the trainings data ######################################################################################

    # 1. check for additional training data
    if os.path.isfile("training_data.npz"):
        self.logger.info("loading additional training data")
        loading = np.load("training_data.npz")
        s_load = loading["s"]
        a_load = loading["a"]
        s_new_load = loading["s_new"]
        r_load = loading["r"]
        r_cumsum_load = loading["r_cumsum"]
        s_new_plus_k_load = loading["s_new_plus_k"]
    else:
        s_load = []
        a_load = []
        s_new_load = []
        r_load = []
        r_cumsum_load = []
        s_new_plus_k_load = []

    # 2. load the data from the last game
    total_reward = 0
    s_list = []
    a_list = []
    s_new_list = []
    r_list = []

    for tuple in self.transitions:
        s, a, s_new, r = tuple
        total_reward += r

        # store all features
        s_list.append(s)
        a_list.append(ACTIONS_DICTIONARY[a])
        r_list.append(r)
        
        # important s_new = None is a special case, here we define an other vector
        if s_new == None:
            # define endstate value, just for data storage
            s_new = torch.zeros(2*n_nearest_coins + (2*field_of_view_size + 1)**2 + 1)
        s_new_list.append(s_new)

    # updating the Network #######################################################################################
    # 1. using the data from the last game
    r_cumsum_list = []
    s_new_plus_k_list = []
    for i in range(len(s_list)):
        s = s_list[i]
        a = a_list[i]
        s_new = s_new_list[i]

        # get cumulated reward
        r_cumsum = np.array(r_list[i: np.min([i + k, len(r_list) - 1])])
        gamma_factor = gamma ** np.arange(len(r_cumsum))
        r_cumsum = np.sum(gamma_factor * r_cumsum)

        s_new_plus_k = s_new_list[np.min([i + k, len(s_new_list) - 1])]

        r_cumsum_list.append(r_cumsum)
        s_new_plus_k_list.append(s_new_plus_k)

        # update Q Function
        if s_new_plus_k[-1] == 0: # end state
            loss = (r_cumsum - self.Q_Network.forward(s)[a])**2
        else:
            loss = (r_cumsum + gamma**k * torch.max(self.Q_Network.forward(s_new_plus_k)) - self.Q_Network.forward(s)[a])**2

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    # 2. using the data from the old games
    if (len(s_load) >= additional_training_set_size) and (len(s_load) != 0):
        batch_indizes = np.random.permutation(len(s_load))[:additional_training_set_size]

    else:
        batch_indizes = np.arange(len(s_load))

    if len(s_load) != 0:
        for i in batch_indizes:
            s = torch.FloatTensor(s_load[i])
            a = a_load[i]
            s_new = torch.FloatTensor(s_new_load[i])
            r = r_load[i]
            r_cumsum = r_cumsum_load[i]
            s_new_plus_k = torch.FloatTensor(s_new_plus_k_load[i])

            # update Q Function
            if s_new_plus_k[-1] == 0:
                loss = (r_cumsum - self.Q_Network.forward(s)[int(a)]) ** 2
            else:
                loss = (r_cumsum + gamma ** k * torch.max(self.Q_Network.forward(s_new_plus_k)) - self.Q_Network.forward(s)[int(a)]) ** 2

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    # Store the model
    self.logger.info("save the Q Network")
    torch.save(self.Q_Network, "DeepQNetwork.pth")


    # export the training data ########################################################################################
    self.logger.info("export training data")

    # concatenate loaded training data and new data
    s_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_load), np.ravel(s_list))).reshape(-1, len(s_list[0])))
    a_all_data = np.concatenate((a_load, a_list))
    s_new_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_new_load), np.ravel(s_new_list))).reshape(-1, len(s_list[0])))
    r_all_data = np.concatenate((r_load, r_list))
    r_cumsum_all_data = np.concatenate((r_cumsum_load, r_cumsum_list))
    s_new_plus_k_all_data = torch.FloatTensor(np.concatenate((np.ravel(s_new_plus_k_load),
                                                              np.ravel(s_new_plus_k_list))).reshape(-1, len(s_list[0])))


    # export data
    np.savez("training_data.npz", s=s_all_data, a=a_all_data, s_new=s_new_all_data, r=r_all_data,
             r_cumsum=r_cumsum_all_data, s_new_plus_k=s_new_plus_k_all_data)


    # export quality data #############################################################################################
    dead = ("KILLED_SELF" in events)
    end_score = last_game_state["self"][1]
    number_of_steps = last_game_state["step"]

    # check if quality data exists:
    if os.path.isfile("training_qualities.npz"):
        loading = np.load("training_qualities.npz")
        dead_load = loading["dead"]
        end_score_load = loading["end_score"]
        number_of_steps_load = loading["number_of_steps"]
        total_reward_load = loading["total_reward"]
    else:
        dead_load = []
        end_score_load = []
        number_of_steps_load = []
        total_reward_load = []

    # restore quality data
    all_dead_data = np.concatenate((dead_load, [dead]))
    all_end_score_data = np.concatenate((end_score_load, [end_score]))
    all_number_of_steps_data = np.concatenate((number_of_steps_load, [number_of_steps]))
    all_total_reward_data = np.concatenate((total_reward_load, [total_reward]))

    np.savez("training_qualities.npz", dead=all_dead_data, end_score=all_end_score_data, number_of_steps=all_number_of_steps_data,
             total_reward=all_total_reward_data)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # good events
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 10,

        # really bad events
        e.INVALID_ACTION: -0.5, # so we learn to navigate and not walking to walls
        e.KILLED_SELF: -10, # this is really bad

        # bad in this context
        e.WAITED: -0.1  # for this task we want to move with minimal number of steps

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
>>>>>>> 2763662b38ee7105bcdd017489e596d0297be8f2
