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
from torch import nn
# python main.py play --no-gui --train 1 --n-rounds 3000 --agents coin_hunter_ag
# ent --scenario coin-heaven

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS_DICTIONARY = dict(zip(ACTIONS, [0, 1, 2, 3, 4, 5]))

# loading hyperparameters
with open('agent_code/QMan/hyperparameters.txt') as f:
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
    s = state_to_features(self, old_game_state)
    s_new = state_to_features(self, new_game_state)

    # get the reward
    r = reward_from_events(self, events, old_game_state, new_game_state)

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
    reward_end = reward_from_events(self, events, last_game_state, None)

    # we want the fastest way:
    reward_end += last_game_state["step"] * -1/(400)
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_end))


    # preparation of the trainings data ######################################################################################

    # 1. check for additional training data
    if os.path.isfile("training_data.npz"):
        self.logger.info("loading additional training data")
        loading = np.load("training_data.npz")
        s_load = loading["s"].tolist()
        a_load = loading["a"]
        s_new_load = loading["s_new"].tolist()
        r_load = loading["r"]

    else:
        s_load = []
        a_load = []
        s_new_load = []
        r_load = []


    # 2. load the data from this game
    total_reward = 0
    s_list = []
    a_list = []
    s_new_list = []
    r_list = []

    for tuple in self.transitions:
        s, a, s_new, r = tuple
        total_reward += r

        # store all features
        s_list.append(s.numpy())
        a_list.append(ACTIONS_DICTIONARY[a])
        r_list.append(r)

        # define some kind of end state
        if s_new == None:
            s_new = torch.zeros_like(s)
        
        s_new_list.append(s_new.numpy())

    # 3. concatenate loaded training data and new data
    s_all_data = torch.FloatTensor(np.array(s_load + s_list))
    a_all_data = np.concatenate((a_load, a_list))
    s_new_all_data = torch.FloatTensor(np.array(s_new_load + s_new_list))
    r_all_data = np.concatenate((r_load, r_list))

    # export the training data ########################################################################################
    self.logger.info("export training data")

    # export data (so we just have replay_buffer_capacity training instances)
    np.savez("training_data.npz", s=s_all_data[-replay_buffer_capacity:], a=a_all_data[-replay_buffer_capacity:],
             s_new=s_new_all_data[-replay_buffer_capacity:], r=r_all_data[-replay_buffer_capacity:])

    # Load old Loss data
    if os.path.isfile("Loss_values.npz"):
        self.logger.info("loading old loss data")
        loading = np.load("Loss_values.npz")
        loss_load = loading["loss"].tolist()
        loading.close()

    else:
        loss_load = []



    # updating the Network #######################################################################################
    # get the minibatch:
    if len(s_all_data[-replay_buffer_capacity:]) == replay_buffer_capacity:
        # we have enough data to train!!!
        self.logger.info("Start training loop for minibatch")

        batch_indizes = np.random.permutation(replay_buffer_capacity)[:batch_size]
        loss = 0
        for index in batch_indizes:
            s = s_all_data[index]
            a = int(a_all_data[index])
            s_new = s_new_all_data[index]
            r = r_all_data[index]

            if torch.all(s_new == 0):  # end state
                loss += (r - self.Q_Network.forward(s)[a]) ** 2
            else:
                loss += (r + gamma * torch.max(self.target_network.forward(s_new)) - self.Q_Network.forward(s)[a]) ** 2

        self.optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Store the loss
        loss_load.append(loss.item())
        np.savez("Loss_values.npz", loss=loss_load)

        # prevent exploding gradients
        nn.utils.clip_grad_norm_(self.Q_Network.parameters(), max_gradient_norm)

        self.optimizer.step()

    # Store the model
    self.logger.info("save the Q Network")
    torch.save(self.Q_Network, "DeepQNetwork.pth")

    # check if we should update the target network:
    if last_game_state["round"] % target_update_steps == 0:
        self.logger.info("Update target network")
        self.target_network = torch.load("DeepQNetwork.pth")

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

    self.logger.info("Export quality data")
    np.savez("training_qualities.npz", dead=all_dead_data, end_score=all_end_score_data, number_of_steps=all_number_of_steps_data,
             total_reward=all_total_reward_data)

    # reset transition store for the next game
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def reward_from_events(self, events: List[str], old_game_state: dict, new_game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # here all rewards are between -1 and 1
    game_rewards = {
        # good events
        e.COIN_COLLECTED:  0.5,
        e.KILLED_OPPONENT: 0.9,
        e.OPPONENT_ELIMINATED: 0.7,
        e.SURVIVED_ROUND: 1,

        # really bad events
        e.INVALID_ACTION: -0.5, # so we learn to navigate and not walking to walls
        e.KILLED_SELF: -1, # this is really bad
        e.GOT_KILLED: -1,

        # bad in this context
        e.WAITED: -0.1  # for this task we want to move with minimal number of steps

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]


    # here we define additional reward ################################################
    if new_game_state != None:
        # our new position
        new_x_position, new_y_position = new_game_state["self"][3]

        # our old position
        old_x_position, old_y_position = old_game_state["self"][3]

        # check if we are in a dangerous position
        dangerous_position = False
        for bomb in new_game_state["bombs"]:
            x_bomb, y_bomb = bomb[0]
            if (np.abs(x_bomb - new_x_position) <= 3) and (y_bomb == new_y_position):
                dangerous_position = True
            if (np.abs(y_bomb - new_y_position) <= 3) and (x_bomb == new_x_position):
                dangerous_position = True

        if dangerous_position:
            reward_sum += -0.3 #FIXME


        # check if we get closer to a coin
        if len(old_game_state["coins"]) != 0:
            old_coin_distances = np.array(manhattan_metric([(old_x_position, old_y_position)], old_game_state["coins"])[0])
            new_coin_distances = np.array(manhattan_metric([(new_x_position, new_y_position)], old_game_state["coins"])[0])

            # important here we have to use the old_game_state for the new_coin_distances, because then the coins already exists
            if any(new_coin_distances < old_coin_distances):
                # we get closer to a coin
                reward_sum += 0.2
            if all(new_coin_distances > old_coin_distances):
                # we went away from the coins
                reward_sum += -0.1



    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
