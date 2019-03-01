import sys
sys.path.append('./classes')
from Agent import Agent
from Robot import Robot
import numpy as np
import FileSystemTools as fst
from RobotTools import RobotTools






run_label = 'Robot__27-02-2019_19-00-16'

rt = RobotTools(resume_dir=fst.combineDirAndFile('/home/pi/robot_car/misc_runs/', run_label))

rt.run()

exit()





rt = RobotTools(
N_total_iterations = 2*10**5,
N_chunks = 200,

features='DQN',
epsilon = 0.97,
eps_decay = 0.9997,
eps_min = 0.05,
N_batch = 100,
N_hidden_layer_nodes = 50,
target_update = 500,
double_DQN = False,
NL_fn = 'tanh',

reward_method = 'software',
debug_enable = 0,
compass_correction_file = '18-02-2019_12-25-37_compass_cal.json'
)

rt.run()

exit()





############ Good for single run, but now trying to use RobotTools.

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
