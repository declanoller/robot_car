import sys
sys.path.append('./classes')

import FileSystemTools as fst
from Agent import Agent
from DummyRobot import DummyRobot

import subprocess as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import torch
import json

############################# Get Robot run dir, if it doesn't already have it.
############# In the future, just check for files it doesn't have.

last_target = 2

remote_hostname = 'pi@192.168.1.240'

#run_dir = 'Robot__22-02-2019_18-15-17'
#run_dir = 'Robot__24-02-2019_20-17-29'
run_dir = 'Robot__27-02-2019_19-00-16'

remote_run_dir_fullpath = fst.combineDirAndFile('/home/pi/robot_car/misc_runs', run_dir)
remote_resume_file = fst.combineDirAndFile(remote_run_dir_fullpath, 'resume.json')
remote_params_file = fst.combineDirAndFile(remote_run_dir_fullpath, 'params.json')
remote_robot_hist_dir = fst.combineDirAndFile(remote_run_dir_fullpath, 'robot_hist')
remote_NN_dir = fst.combineDirAndFile(remote_run_dir_fullpath, 'NN_info')

local_base_dir = 'misc_runs/'
local_final_dir = fst.combineDirAndFile(local_base_dir, run_dir)
robot_hist_dir = fst.combineDirAndFile(local_final_dir, 'robot_hist')
NN_dir = fst.combineDirAndFile(local_final_dir, 'NN_info')

resume_file = fst.combineDirAndFile(local_final_dir, 'resume.json')
params_file = fst.combineDirAndFile(local_final_dir, 'params.json')

if os.path.exists(local_final_dir):
    print('\n\n{} exists already, skipping for now.'.format(local_final_dir))
else:

    print('\ncreating dirs:')
    print(local_final_dir)
    print(robot_hist_dir)
    print(NN_dir)
    # Make dirs
    fst.makeDir(local_final_dir)
    fst.makeDir(robot_hist_dir)
    fst.makeDir(NN_dir)


#scp_cmd = ['scp', '-r', '{}:{}'.format(remote_hostname, remote_run_dir_fullpath), local_base_dir]
scp_cmd = ['scp', '{}:{}'.format(remote_hostname, remote_resume_file), local_final_dir]
print('\n\nCalling scp command to retrieve resume.json...\n\n')
sp.check_call(scp_cmd)
print('\n\nFiles transferred.\n')

scp_cmd = ['scp', '{}:{}'.format(remote_hostname, remote_params_file), local_final_dir]
print('\n\nCalling scp command to retrieve params.json...\n\n')
sp.check_call(scp_cmd)
print('\n\nFiles transferred.\n')

# Get the resume info
with open(resume_file, 'r') as f:
    resume_info = json.load(f)

# Get the params info
with open(params_file, 'r') as f:
    params_info = json.load(f)

chunks_completed = resume_info['next_starting_chunk']
last_chunk_index = chunks_completed - 1
iter_per_chunk = params_info['N_iterations_per_chunk']

################################ Read in robot_hist, plot

# Now this will only get the ones it hasn't yet

robot_hist_datlist = []

for chunk in range(chunks_completed):

    robot_hist_file = 'robot_hist_chunk_{}_iterations_{}-{}.txt'.format(chunk, chunk*iter_per_chunk, (chunk + 1)*iter_per_chunk - 1)
    robot_hist_file_fullpath = fst.combineDirAndFile(robot_hist_dir, robot_hist_file)

    if not os.path.exists(robot_hist_file_fullpath):
        print('dont have robot_hist file {}, calling scp to retrieve'.format(robot_hist_file))
        remote_robot_hist_file = fst.combineDirAndFile(remote_robot_hist_dir, robot_hist_file)
        scp_cmd = ['scp', '{}:{}'.format(remote_hostname, remote_robot_hist_file), robot_hist_dir]
        sp.check_call(scp_cmd)

    robot_hist_datlist.append(np.loadtxt(robot_hist_file_fullpath))


print('\nShape of first robot_hist file: ', robot_hist_datlist[0].shape)
robot_hist = np.concatenate(robot_hist_datlist)
print('\nShape of combined robot_hist dat: ', robot_hist.shape)




########################## Get latest NN


last_NN_model_file = 'NN_model_chunk_{}_iterations_{}-{}.txt'.format(last_chunk_index, last_chunk_index*iter_per_chunk, (last_chunk_index + 1)*iter_per_chunk - 1)
last_NN_model_file_fullpath = fst.combineDirAndFile(NN_dir, last_NN_model_file)

if not os.path.exists(last_NN_model_file_fullpath):
    print('dont have last NN_model file {}, calling scp to retrieve'.format(last_NN_model_file))
    remote_NN_model_file = fst.combineDirAndFile(remote_NN_dir, last_NN_model_file)
    scp_cmd = ['scp', '{}:{}'.format(remote_hostname, remote_NN_model_file), NN_dir]
    sp.check_call(scp_cmd)


########################### Plot some stuff

## Columns currently are: [iter	t	x	y	ang	r	action	target]

fig = plt.figure(figsize=(15,9))

gs = GridSpec(4, 6, figure=fig)

ax_xy = fig.add_subplot(gs[0,0])
ax_R = fig.add_subplot(gs[2,0])
ax_R_recent = fig.add_subplot(gs[2,1])
ax_R_avg = fig.add_subplot(gs[3,0])
ax_R_window_avg = fig.add_subplot(gs[3,1])
ax_ang = fig.add_subplot(gs[0,1])
ax_action = fig.add_subplot(gs[1,1])

axes_Qvals = [
fig.add_subplot(gs[0,2]),
fig.add_subplot(gs[1,2]),
fig.add_subplot(gs[2,2]),
fig.add_subplot(gs[3,2]),
fig.add_subplot(gs[0,3]),
fig.add_subplot(gs[1,3]),
fig.add_subplot(gs[2,3]),
fig.add_subplot(gs[3,3]),
]


axes_best_act = [
fig.add_subplot(gs[0,4]),
fig.add_subplot(gs[1,4]),
fig.add_subplot(gs[2,4]),
fig.add_subplot(gs[3,4]),
fig.add_subplot(gs[0,5]),
fig.add_subplot(gs[1,5]),
fig.add_subplot(gs[2,5]),
fig.add_subplot(gs[3,5]),
]


N_most_recent = 1000

ax_xy.set_xlabel('iteration')
ax_xy.set_ylabel('x, y')
ax_xy.plot(robot_hist[-N_most_recent:,0], robot_hist[-N_most_recent:, 2])
ax_xy.plot(robot_hist[-N_most_recent:,0], robot_hist[-N_most_recent:, 3])

ax_ang.set_xlabel('iteration')
ax_ang.set_ylabel('angle')
ax_ang.plot(robot_hist[-N_most_recent:,0], robot_hist[-N_most_recent:, 4])


ax_action.set_xlabel('iteration')
ax_action.set_ylabel('action')
ax_action.plot(robot_hist[-N_most_recent:,0], robot_hist[-N_most_recent:, 6])


ax_R.set_xlabel('iteration')
ax_R.set_ylabel('R')
ax_R.plot(robot_hist[:,0], robot_hist[:, 5])


ax_R_recent.set_xlabel('iteration')
ax_R_recent.set_ylabel('R_last_{}'.format(N_most_recent))
ax_R_recent.plot(robot_hist[-N_most_recent:,0], robot_hist[-N_most_recent:, 5])
ax_R_recent.axhline(robot_hist[-N_most_recent:, 5].mean(), linestyle='dashed', color='darkred')


N_start = 400

ax_R_avg.set_xlabel('iteration')
ax_R_avg.set_ylabel('R_avg')
ax_R_avg.plot(robot_hist[N_start:,0], np.cumsum(robot_hist[N_start:, 5])/(1 + robot_hist[N_start:,0]))



def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


N_start = 400

N_window = 1000

ax_R_window_avg.set_xlabel('iteration')
ax_R_window_avg.set_ylabel('R_{}_avg'.format(N_window))

window_avg = running_mean(robot_hist[N_start:, 5], N_window)

ax_R_window_avg.plot(list(range(len(window_avg))), window_avg)





##################################### Q values as a function of position

N_disc = 100
wall_length = 1.25
arena_lims = np.array([-wall_length/2, wall_length/2])

target_positions = np.array([
[.19, 0],
[.55, 0],
[wall_length, .21],
[wall_length, .65],
[.97, wall_length],
[.58, wall_length],
[0, 1.02],
[0, .60]])

N_targets = len(target_positions)
target_positions = target_positions + arena_lims[0]


# Ugh I think I actually have to create an Agent until I do it in a more separated way...

ag = Agent(
agent_class=DummyRobot,
features='DQN',
N_steps=5*10**4,
N_hidden_layer_nodes = 50,
double_DQN = False,
NL_fn = 'tanh',
save_params_to_file = False,
)

ag.loadModel(fname=last_NN_model_file_fullpath)


pos = np.array([[[x,y] for y in np.linspace(*arena_lims, N_disc)] for x in np.linspace(*arena_lims, N_disc)])
ang = np.expand_dims(np.full((N_disc,N_disc), 0), axis=2)

for target in range(N_targets):

    last_target_pos = target_positions[target]
    #print('last target pos:', last_target_pos)
    # This is creating an array of tensors that we can feed into the trained NN to see its
    # current estimate of the Q function, etc. It uses angle=0, and the last target, and then
    # plots Q for all the pos.

    targ_pos_x = np.expand_dims(np.full((N_disc,N_disc), last_target_pos[0]), axis=2)
    targ_pos_y = np.expand_dims(np.full((N_disc,N_disc), last_target_pos[1]), axis=2)

    states = np.concatenate((pos, ang, targ_pos_x, targ_pos_y), axis=2)
    states = torch.tensor(states, dtype=torch.float32)

    NN_output = ag.forwardPass(ag.policy_NN, states)

    max_Q = (torch.max(NN_output, dim=2)[0]).detach().numpy()
    best_actions = (torch.argmax(NN_output, dim=2)).detach().numpy()

    ######### Plot Q vals

    ax_Q = axes_Qvals[target    ]

    col_plot_Q = ax_Q.matshow(max_Q.T, cmap='Reds', origin='lower')

    ax_Q.set_xlabel('x')
    ax_Q.set_ylabel('y')
    ax_Q.set_xticks([])
    ax_Q.set_yticks([])
    ax_Q.set_xticklabels([])
    ax_Q.set_yticklabels([])

    target_rad = N_disc/17
    # The thing is plotted in terms of the indices of max_Q_Q, not the actual x and y vals...
    target_circle = plt.Circle(
    ((last_target_pos[0] - arena_lims[0])*N_disc/wall_length, (last_target_pos[1] - arena_lims[0])*N_disc/wall_length),
    target_rad, color='black')

    ax_Q.add_artist(target_circle)

    ######## Plot best actions

    ax_best_act = axes_best_act[target]

    cm = LinearSegmentedColormap.from_list('my_cm', ['tomato','dodgerblue','seagreen','orange'], N=4)
    col_plot_bestact = ax_best_act.matshow(best_actions.T, cmap=cm, origin='lower')

    ax_best_act.set_xlabel('x')
    ax_best_act.set_ylabel('y')
    ax_best_act.set_xticks([])
    ax_best_act.set_yticks([])
    ax_best_act.set_xticklabels([])
    ax_best_act.set_yticklabels([])

    # The thing is plotted in terms of the indices of max_Q, not the actual x and y vals...
    target_circle = plt.Circle(
    ((last_target_pos[0] - arena_lims[0])*N_disc/wall_length, (last_target_pos[1] - arena_lims[0])*N_disc/wall_length),
    target_rad, color='black')

    col_bar = fig.colorbar(col_plot_bestact, ax=ax_best_act, ticks=[0,1,2,3], boundaries=np.arange(-.5,4.5,1))
    col_bar.ax.set_yticklabels(['F','B','CCW','CW'])

    ax_best_act.add_artist(target_circle)



plt.tight_layout()

plt.savefig(fst.combineDirAndFile(local_final_dir, run_dir + '.png'))

plt.show()












exit()














#
