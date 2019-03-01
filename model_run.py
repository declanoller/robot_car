import sys
sys.path.append('./classes')
from Agent import Agent
from Robot import Robot
import FileSystemTools as fst




################ BEST MODEL FOR SIMULATING

ag = Agent(agent_class=Robot, features='DQN', N_steps=2*10**5,
epsilon = 0.9,
eps_decay = 0.999,
eps_min = 0.1,
N_batch = 80,
N_hidden_layer_nodes = 50,
target_update = 500,
double_DQN = False,
NL_fn = 'tanh',
)

ag.DQNepisode(show_plot=False, save_plot=True)

exit(0)


################ BEST MODEL FOR LOADING

path = '/home/pi/robot_car/saved_models/no_v_400k'
model = 'model_Agent=PuckworldAgent_robot_epsilon=0.90000_eps_decay=0.99950_N_steps=400000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=False_features=DQN_NL_fn=tanh_loss_method=L2_advantage=True_beta=0.10000_22-29-32.model'
model_fname = fst.combineDirAndFile(path, model)
ag.loadModelPlay(model_fname, show_plot=False, load_params=False, zero_eps=False)

exit(0)

path = '/home/pi/robot_car/saved_models/random, drag, small targets radial'
model = 'model_Agent=PuckworldAgent_radial_epsilon=0.900_eps_decay=1.000_N_steps=400000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=False_features=DQN_NL_fn=tanh_loss_method=L2_advantage=True_beta=0.100_14-46-05.model'
model_fname = fst.combineDirAndFile(path, model)
ag.loadModelPlay(model_fname, show_plot=True, load_params=False, zero_eps=False)

exit(0)














#
