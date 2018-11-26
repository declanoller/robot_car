#import AgentTools as at
from Agent import Agent

from Robot import Robot
import numpy as np
import FileSystemTools as fst


ag = Agent(agent_class=Robot, features='DQN', N_steps=4*10**5,
epsilon = 0.9,
eps_decay = 0.9995,
eps_min = 0.05,
N_batch = 80,
N_hidden_layer_nodes = 50,
target_update = 500,
double_DQN = False,
NL_fn = 'tanh',
motor_enable=True, sonar_enable=True, compass_enable=True, MQTT_enable=True
)


exit(0)


path = '/home/pi/robot_car/saved_models/random, drag, small targets radial'
model = 'model_Agent=PuckworldAgent_radial_epsilon=0.900_eps_decay=1.000_N_steps=400000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=False_features=DQN_NL_fn=tanh_loss_method=L2_advantage=True_beta=0.100_14-46-05.model'
model_fname = fst.combineDirAndFile(path, model)
ag.loadModelPlay(model_fname, show_plot=True, load_params=False, zero_eps=True)















#
