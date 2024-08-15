Hyperparameters:

The hyperparameters are in the txt file and are loaded into the code from there. The names of the hyperparameters in the file then correspond to the variable names.

n_nearest_coins corresponds to the number of next coins to be tracked. The x and y differences of each next coin are saved as a feature so that the character knows
in which direction it would have to go and for how long to reach the coin

epsilon and epsilon_decay are used for the epsilon-greedy policy.

additional_training_set_size defines how much additional training data from past games should be used as training data in each game

field_of_view_size: as a second feature, my network uses a kind of local field, i.e. the closest surroundings of the character. The idea is that in the event of bombs and explosions, the character
only needs to know those in the immediate vicinity, and the character should recognize safe fields in the vicinity so that it can take cover from bombs.
The field_of_view_size hyperparameter therefore defines the radius of the local environment

k: this parameter describes how many states we should look further in the loss function, this serves to improve the loss function.

max_gradient_norm: In my case, it happened that the gradients exploded, which is why I limit the gradients here.

use_rule_based_agent: This decides whether to play with the rule-based agent. It then plays the game and incidentally collects training data to train our model.
The idea is that we should learn how the rule based agent works.



The code of the agent:
The collected features are coded in a tensor. For each next coin, the relative x and y differences (direction and distance of the coin) are stored and 
a local environment is stored to recognize escape routes and bombs in the environment.

The neural network is stored in a .pth file, the collected training data is stored in a .npz file, the existing training data is reloaded and used again and again during training.

Other features, such as whether you have died at the end, are saved in another .npz file. There is also a Jupyter Notebook “Evaluation” included, with which you can evaluate this
data and monitor the suicide rate, for example.


