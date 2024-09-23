from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features
from .callbacks import feature_vector_to_state_index
from .callbacks import ACTIONS
from .callbacks import n_nearest_bombs, n_nearest_coins
import os
# python main.py play --no-gui --train 1 --n-rounds 3000 --agents coin_hunter_ag
# ent --scenario coin-heaven

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS_DICTIONARY = dict(zip(ACTIONS, [0, 1, 2, 3, 4, 5]))

# Hyper parameters
learning_rate = 0.2
gamma = 0.9
additional_training_set_size = 100 # we use also 100 instances of the old training data
k = 4 # for k-step Q-learning, the bombs explode in 4 steps, therefore k = 4 is the minimum

TRANSITION_HISTORY_SIZE = 100



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
    s_vector = state_to_features(old_game_state)
    s = feature_vector_to_state_index(s_vector)

    s_new_vector = state_to_features(new_game_state)
    s_new = feature_vector_to_state_index(s_new_vector)

    # get the reward
    r = reward_from_events(self, events)

    # look if we get closer to a bomb
    old_bomb_distances = s_vector[n_nearest_bombs + n_nearest_coins + 4:]
    new_bomb_distances = s_new_vector[n_nearest_bombs + n_nearest_coins + 4:]
    
    if np.any(new_bomb_distances < old_bomb_distances and (new_bomb_distances != -1) and (old_bomb_distances != -1)):
        r -= 0.5
    if np.all(new_bomb_distances > old_bomb_distances or (new_bomb_distances == -1)):
        r += 0.3

    # restore the states, actions and rewards
    self.transitions.append(Transition(s_vector, self_action, s_new_vector, r))
    self.transitions_state_indizes.append(Transition(s, ACTIONS_DICTIONARY[self_action], s_new, r))

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
    self.transitions_state_indizes.append(
        Transition(feature_vector_to_state_index(state_to_features(last_game_state)), ACTIONS_DICTIONARY[last_action],
                   None, reward_from_events(self, events)))


    # if additional training data exists, we use it
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

    # loop over all training instances of the last game
    total_reward = 0
    s_list = []
    a_list = []
    s_new_list = []
    r_list = []

    for tuple in self.transitions_state_indizes:
        s, a, s_new, r = tuple
        total_reward += r

        if s_new == None:
            s_new = -1

        # store all features for the future
        s_list.append(s)
        a_list.append(a)
        s_new_list.append(s_new)
        r_list.append(r)

    # updating using last game ###################################################################################
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

        s_new_plus_k = int(s_new_list[np.min([i + k, len(s_new_list) - 1])])

        r_cumsum_list.append(r_cumsum)
        s_new_plus_k_list.append(s_new_plus_k)

        # update Q Function
        self.Q[s, a] = self.Q[s, a] + learning_rate* (r_cumsum + gamma**k * np.max(self.Q[s_new_plus_k, :]) - self.Q[s, a])

    # updating using old tmaxing_set_size) and (len(s_load) != 0):
        batch_indizes = np.random.permutation(len(s_load))[:additional_training_set_size]

        for i in batch_indizes:

            s = int(s_load[i])
            a = int(a_load[i])
            s_new = int(s_new_load[i])
            r = r_load[i]
            r_cumsum = r_cumsum_load[i]
            s_new_plus_k = int(s_new_plus_k_load[i])
        
            # update Q function
            self.Q[s, a] = self.Q[s, a] + learning_rate * (r_cumsum + gamma ** k * np.max(self.Q[s_new_plus_k, :]) - self.Q[s, a])

    # Store the model
    self.logger.info("save the new Q matrix")
    np.save("Q_saved.npy", self.Q)

    # export the training data ###################################
    self.logger.info("export training data")

    # concatenate loaded training data and new data
    s_all_data = np.concatenate((s_load, s_list))
    a_all_data = np.concatenate((a_load, a_list))
    s_new_all_data = np.concatenate((s_new_load, s_new_list))
    r_all_data = np.concatenate((r_load, r_list))
    r_cumsum_all_data = np.concatenate((r_cumsum_load, r_cumsum_list))
    s_new_plus_k_all_data = np.concatenate((s_new_plus_k_load, s_new_plus_k_list))

    # export data
    np.savez("training_data.npz", s=s_all_data, a=a_all_data, s_new=s_new_all_data, r=r_all_data,
             r_cumsum=r_cumsum_all_data, s_new_plus_k=s_new_plus_k_all_data)

    #with open('training_data.pkl', 'ab') as the_file:
    #for state in self.transitions:
    #    s, a, s_new, r = state
    #    if s_new != None:
    #        np.save("training_data", s + [a] + s_new + [r])
    #    else:
    #        np.save("training_data", s + [a] + [s_new] + [r])
            #pickle.dump([s, a, s_new, r], the_file)
    #with open('training_data_state_indizes.pkl', 'ab') as the_file:
    #for state in self.transitions_state_indizes:
    #    s, a, s_new, r = state
    #    np.save("training_data_state_indizes", [s, a, s_new, r])
    #        pickle.dump([s, a, s_new, r], the_file)

    # to look at the quality of our model we save some additional informations
    dead = ("KILLED_SELF" in events)
    end_score = last_game_state["self"][1]
    number_of_steps = last_game_state["step"]
    Q_norm = np.linalg.norm(self.Q, np.inf)

    # check if quality data exists:
    if os.path.isfile("training_qualities.npz"):
        loading = np.load("training_qualities.npz")
        dead_load = loading["dead"]
        end_score_load = loading["end_score"]
        number_of_steps_load = loading["number_of_steps"]
        Q_norm_load = loading["Q_norm"]
        total_reward_load = loading["total_reward"]
    else:
        dead_load = []
        end_score_load = []
        number_of_steps_load = []
        Q_norm_load = []
        total_reward_load = []

    # restore quality data
    all_dead_data = np.concatenate((dead_load, [dead]))
    all_end_score_data = np.concatenate((end_score_load, [end_score]))
    all_number_of_steps_data = np.concatenate((number_of_steps_load, [number_of_steps]))
    all_Q_norm_data = np.concatenate((Q_norm_load, [Q_norm]))
    all_total_reward_data = np.concatenate((total_reward_load, [total_reward]))

    np.savez("training_qualities.npz", dead=all_dead_data, end_score=all_end_score_data, number_of_steps=all_number_of_steps_data,
             Q_norm=all_Q_norm_data, total_reward=all_total_reward_data)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # good events
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 10,

        # really bad events
        e.INVALID_ACTION: -0.5, # so we learn to navigate and not walking to walls
        e.KILLED_SELF: -10, # this is really bad

        # bad in this context
        e.WAITED: -1  # for this task we want to move with minimal number of steps

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
