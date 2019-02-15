from RobotTools import RobotTools
#from Robot import Robot
#agent_class=Robot,



rt = RobotTools(resume_dir='/home/pi/robot_car/misc_runs/Robot__15-02-2019_15-57-31')

rt.run()

exit()



rt = RobotTools(
N_total_iterations=40,
N_chunks=5,

features='DQN',
epsilon = 0.9,
eps_decay = 0.999,
eps_min = 0.05,
N_batch = 100,
N_hidden_layer_nodes = 50,
target_update = 500,
double_DQN = False,
NL_fn = 'tanh',

compass_correction_file='13-02-2019_13-44-57_compass_cal.json'
)

rt.run()

exit()


#
