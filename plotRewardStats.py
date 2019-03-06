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
import argparse

############################# Get Robot run dir, if it doesn't already have it.
############# In the future, just check for files it doesn't have.

run_dir = 'Robot__02-03-2019_14-19-38'

local_base_dir = 'misc_runs/'
local_final_dir = fst.combineDirAndFile(local_base_dir, run_dir)
robot_hist_dir = fst.combineDirAndFile(local_final_dir, 'robot_hist')
NN_dir = fst.combineDirAndFile(local_final_dir, 'NN_info')

resume_file = fst.combineDirAndFile(local_final_dir, 'resume.json')
params_file = fst.combineDirAndFile(local_final_dir, 'params.json')

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
    robot_hist_datlist.append(np.loadtxt(robot_hist_file_fullpath))


print('\nShape of first robot_hist file: ', robot_hist_datlist[0].shape)
robot_hist = np.concatenate(robot_hist_datlist)
print('\nShape of combined robot_hist dat: ', robot_hist.shape)



############## Plotting

fig = plt.figure(figsize=(12,8))

gs = GridSpec(2, 3, figure=fig)

ax_R = fig.add_subplot(gs[0,0])
ax_R_recent = fig.add_subplot(gs[0,1])
ax_R_avg = fig.add_subplot(gs[1,0])
ax_R_window_avg = fig.add_subplot(gs[1,1])

ax_R_delay = fig.add_subplot(gs[0,2])


N_most_recent = 1000

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


N_window = 1000

ax_R_window_avg.set_xlabel('iteration')
ax_R_window_avg.set_ylabel('R_{}_avg'.format(N_window))

window_avg = running_mean(robot_hist[N_start:, 5], N_window)

ax_R_window_avg.plot(list(range(len(window_avg))), window_avg)


# This counts the time between getting rewards.
delay_count = 0
cur_target = robot_hist[0,7]
delay_list = []
#for i in range(0, int(robot_hist[3*iter_per_chunk, 0])):
for i in range(int(robot_hist[-5*iter_per_chunk, 0]), int(robot_hist[-1, 0])):

    if robot_hist[i, 7] != cur_target:
        cur_target = robot_hist[i, 7]
        #delay_list.append(delay_count)
        delay_list += [delay_count]*delay_count
        delay_count = 0
    else:
        delay_count += 1


ax_R_delay.hist(delay_list)









plt.show()














#
