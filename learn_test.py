from Agent import Agent

from Robot import Robot
import numpy as np
import FileSystemTools as fst


ag = Agent(agent_class=Robot, features='DQN', N_steps=5*10**4,
epsilon = 0.9,
eps_decay = 0.999,
eps_min = 0.05,
N_batch = 100,
N_hidden_layer_nodes = 50,
target_update = 500,
double_DQN = False,
NL_fn = 'tanh',

save_hist=False,
compass_correction_file='13-02-2019_13-44-57_compass_cal.json',
reward_method='calculated'
)

ag.DQNepisode(show_plot=False, save_plot=True)

























#
