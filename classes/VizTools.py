import sys
sys.path.append('./classes')
sys.path.append('.')

import FileSystemTools as fst
from Agent import Agent
from DummyRobot import DummyRobot_5state, DummyRobot_6state
import movieMaker

import traceback as tb
import time
import os
import numpy as np
import torch
import json
import argparse
import subprocess as sp
from math import tan, sin, cos, pi

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import ImageFile,Image,ImageDraw
from mpl_toolkits.axes_grid1 import make_axes_locatable

class VizTools:


    def __init__(self):

        self.remote_hostname = 'pi@192.168.1.240'
        self.latest_loaded = False



    def loadAll(self, run_dir, **kwargs):
        # Fetches the most recent data from the run_dir passed to it on the robot,
        # unless you tell it to just use what it has.
        #
        nofetch = kwargs.get('nofetch', True)

        print('loading all, with nofetch = {}'.format(nofetch))

        self.dirVarsSetup(run_dir) # Create the member vars

        if nofetch:
            assert self.checkRunDirExists(), 'nofetch = True; Local run dir must already exist if not fetching data!'

        self.dirSetup() # Create the dirs if they don't exist yet

        if not nofetch:
            self.fetchUpdate()

        print('loading run info')
        self.loadRunInfo()
        print('loading robot hist')
        self.loadRobotHist()
        #print('getting NN')
        self.fetchNNInfo(**kwargs)
        last_NN_model_file = 'NN_model_chunk_{}_iterations_{}-{}.txt'.format(
        self.last_chunk_index, self.last_chunk_index*self.iter_per_chunk, (self.last_chunk_index + 1)*self.iter_per_chunk - 1)

        self.last_NN_model_file_fullpath = fst.combineDirAndFile(self.NN_dir, last_NN_model_file)


    def plotProgress(self, run_dir, **kwargs):

        # Then plots it.
        if not self.latest_loaded:
            self.loadAll(run_dir, **kwargs)
            self.latest_loaded = True

        self.plotAll(**kwargs)
        self.plotCurves(**kwargs)

        self.plotQ(**kwargs)
        self.plotActions(**kwargs)



    def createQVid(self, run_dir, **kwargs):

        # This will plot Q for the different NN's over time, and create a gif
        if not self.latest_loaded:
            self.loadAll(run_dir, fetch_range='all', **kwargs)
            self.latest_loaded = True

        Q_plot_dir = os.path.join(self.local_final_dir, 'Q_plots')
        if not os.path.exists(Q_plot_dir):
            fst.makeDir(Q_plot_dir)

        act_plot_dir = os.path.join(self.local_final_dir, 'act_plots')
        if not os.path.exists(act_plot_dir):
            fst.makeDir(act_plot_dir)

        NN_model_file_template = os.path.join(self.NN_dir, 'NN_model_chunk_{}_iterations_{}-{}.txt')

        for chunk in range(self.chunks_completed):

            NN_model_file = NN_model_file_template.format(
            chunk, chunk*self.iter_per_chunk, (chunk + 1)*self.iter_per_chunk - 1)

            save_fname = NN_model_file.replace('.txt', '')

            plot_title = 'Iterations {}-{}.txt'.format(chunk*self.iter_per_chunk, (chunk + 1)*self.iter_per_chunk - 1)

            if os.path.exists(NN_model_file):
                self.plotQ(
                save_fname=os.path.join(Q_plot_dir, f'{chunk}.png'),
                plot_title=plot_title,
                NN_model_fname=NN_model_file,
                noshow=True)

                self.plotActions(
                save_fname=os.path.join(act_plot_dir, f'{chunk}.png'),
                plot_title=plot_title,
                NN_model_fname=NN_model_file,
                noshow=True)

        im = Image.open(os.path.join(Q_plot_dir, '0.png'))
        res = '{}x{}'.format(*(im.size))
        movieMaker.imgsToVid(Q_plot_dir, os.path.join(self.local_final_dir, 'Q_fn.mp4'), framerate=20, crf=15, res=res)
        movieMaker.imgsToVid(act_plot_dir, os.path.join(self.local_final_dir, 'opt_act.mp4'), framerate=20, crf=15, res=res)




    def fetchUpdate(self):

        # Should probably assert here to check that var are defined

        self.fetchRunInfo()
        self.loadRunInfo()
        self.fetchLogs()


    def fetchRunInfo(self):

        scp_cmd = ['scp', '{}:{}'.format(self.remote_hostname, self.remote_resume_file), self.local_final_dir]
        print('\n\nCalling scp command to retrieve resume.json...\n\n')
        sp.check_call(scp_cmd)
        print('\n\nFiles transferred.\n')

        scp_cmd = ['scp', '{}:{}'.format(self.remote_hostname, self.remote_params_file), self.local_final_dir]
        print('\n\nCalling scp command to retrieve params.json...\n\n')
        sp.check_call(scp_cmd)
        print('\n\nFiles transferred.\n')


    def loadRunInfo(self):

        # Get the resume info
        with open(self.resume_file, 'r') as f:
            self.resume_info = json.load(f)

        # Get the params info
        with open(self.params_file, 'r') as f:
            self.params_info = json.load(f)

        self.chunks_completed = self.resume_info['next_starting_chunk']
        self.last_chunk_index = self.chunks_completed - 1
        self.iter_per_chunk = self.params_info['N_iterations_per_chunk']
        self.latest_str = '_chunk_{}_iterations_{}-{}'.format(
        self.last_chunk_index, self.last_chunk_index*self.iter_per_chunk, (self.last_chunk_index + 1)*self.iter_per_chunk - 1)


    def fetchLogs(self):

        self.fetchRobotHist()
        self.fetchNNInfo()


    def fetchRobotHist(self):


        ################################ Read in robot_hist, plot

        # Now this will only get the ones it hasn't yet

        robot_hist_datlist = []

        for chunk in range(self.chunks_completed):

            robot_hist_file = 'robot_hist_chunk_{}_iterations_{}-{}.txt'.format(
            chunk, chunk*self.iter_per_chunk, (chunk + 1)*self.iter_per_chunk - 1)
            robot_hist_file_fullpath = fst.combineDirAndFile(self.robot_hist_dir, robot_hist_file)

            if not os.path.exists(robot_hist_file_fullpath):
                print('dont have robot_hist file {}, calling scp to retrieve'.format(robot_hist_file))
                remote_robot_hist_file = fst.combineDirAndFile(self.remote_robot_hist_dir, robot_hist_file)
                scp_cmd = ['scp', '{}:{}'.format(self.remote_hostname, remote_robot_hist_file), self.robot_hist_dir]
                sp.check_call(scp_cmd)


    def loadRobotHist(self):


        ################################ Read in robot_hist, plot

        # Now this will only get the ones it hasn't yet

        robot_hist_datlist = []

        robot_hist_file_template = os.path.join(self.robot_hist_dir, 'robot_hist_chunk_{}_iterations_{}-{}.txt')

        for chunk in range(self.chunks_completed):

            robot_hist_file = robot_hist_file_template.format(
            chunk,
            chunk*self.iter_per_chunk,
            (chunk + 1)*self.iter_per_chunk - 1)

            robot_hist_datlist.append(np.loadtxt(robot_hist_file))

        print('\nShape of first robot_hist file: ', robot_hist_datlist[0].shape)
        self.robot_hist = np.concatenate(robot_hist_datlist)
        print('\nShape of combined robot_hist dat: ', self.robot_hist.shape)

        first_hist_fname = robot_hist_file_template.format(
        0,
        0*self.iter_per_chunk,
        (0 + 1)*self.iter_per_chunk - 1)

        with open(first_hist_fname) as f:
            first_line = f.readline().split('\t')

        self.history_columns = [x.replace('#','').strip() for x in first_line]
        print('Columns of robot_hist: ', self.history_columns)


    def fetchNNInfo(self, **kwargs):

        fetch_range = kwargs.get('fetch_range', 'latest')
        assert fetch_range in ['latest', 'all'], 'fetch_range but be either latest or all'

        if fetch_range == 'latest':

            last_NN_model_file = 'NN_model_chunk_{}_iterations_{}-{}.txt'.format(
            self.last_chunk_index, self.last_chunk_index*self.iter_per_chunk, (self.last_chunk_index + 1)*self.iter_per_chunk - 1)

            self.last_NN_model_file_fullpath = fst.combineDirAndFile(self.NN_dir, last_NN_model_file)

            if not os.path.exists(self.last_NN_model_file_fullpath):
                print('dont have last NN_model file {}, calling scp to retrieve'.format(last_NN_model_file))
                remote_NN_model_file = fst.combineDirAndFile(self.remote_NN_dir, last_NN_model_file)
                scp_cmd = ['scp', '{}:{}'.format(self.remote_hostname, remote_NN_model_file), self.NN_dir]
                sp.check_call(scp_cmd)

        if fetch_range == 'all':

            for chunk in range(self.chunks_completed):

                NN_model_file = 'NN_model_chunk_{}_iterations_{}-{}.txt'.format(
                chunk, chunk*self.iter_per_chunk, (chunk + 1)*self.iter_per_chunk - 1)

                NN_model_file_fullpath = fst.combineDirAndFile(self.NN_dir, NN_model_file)

                if not os.path.exists(NN_model_file_fullpath):
                    print('dont have NN_model file {}, calling scp to retrieve'.format(NN_model_file))
                    remote_NN_model_file = fst.combineDirAndFile(self.remote_NN_dir, NN_model_file)
                    scp_cmd = ['scp', '{}:{}'.format(self.remote_hostname, remote_NN_model_file), self.NN_dir]
                    sp.check_call(scp_cmd)




########################### Plotting fns


    def plotQ(self, **kwargs):

        fig = plt.figure(figsize=(8.4,4.8))
        gs = GridSpec(2, 4, figure=fig)

        axes_Qvals = [
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[0,3]),
        fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[1,1]),
        fig.add_subplot(gs[1,2]),
        fig.add_subplot(gs[1,3]),
        ]

        title = kwargs.get('plot_title', '')
        save_fname = kwargs.get('save_fname', os.path.join(self.local_final_dir,'plot_{}_Q.png'.format(self.latest_str)))

        self.plotQAndActions(axes_Qvals, None, **kwargs, plot_Q=True, plot_act=False)
        plt.suptitle(title)
        plt.tight_layout()

        plt.savefig(save_fname)

        if not kwargs.get('noshow', False):
            plt.show()
        plt.close('all')



    def plotActions(self, **kwargs):

        fig = plt.figure(figsize=(8.4,4.8))
        gs = GridSpec(2, 4, figure=fig)

        axes_best_act = [
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[0,3]),
        fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[1,1]),
        fig.add_subplot(gs[1,2]),
        fig.add_subplot(gs[1,3]),
        ]

        title = kwargs.get('plot_title', '')
        save_fname = kwargs.get('save_fname', os.path.join(self.local_final_dir,'plot_{}_act.png'.format(self.latest_str)))

        self.plotQAndActions(None, axes_best_act, **kwargs, plot_Q=False, plot_act=True)
        plt.suptitle(title)
        plt.tight_layout(pad=0.3)

        plt.savefig(save_fname)

        if not kwargs.get('noshow', False):
            plt.show()
        plt.close('all')



    def plotAll(self, **kwargs):

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

        self.plotXY(ax_xy)
        self.plotAng(ax_ang)
        self.plotAction(ax_action)
        self.plotAllR(ax_R)
        self.plotRecentR(ax_R_recent)
        self.plotAvgR(ax_R_avg)
        self.plotRecentAvgR(ax_R_window_avg)

        #self.plotQAndActions(axes_Qvals, axes_best_act)

        plt.tight_layout()

        plt.savefig(fst.combineDirAndFile(self.local_final_dir, 'plot_{}_all.png'.format(self.latest_str)))
        if not kwargs.get('noshow', False):
            plt.show()

        plt.close('all')




    def plotCurves(self, **kwargs):

        ########################### Plot some stuff

        ## Columns currently are: [iter	t	x	y	ang	r	action	target]

        fig = plt.figure(figsize=(8.4,4.8))

        gs = GridSpec(2, 3, figure=fig)

        ax_xy = fig.add_subplot(gs[0,0])
        ax_ang = fig.add_subplot(gs[0,1])
        ax_action = fig.add_subplot(gs[0,2])

        ax_R_recent = fig.add_subplot(gs[1,2])
        ax_R_avg = fig.add_subplot(gs[1,0])
        ax_R_window_avg = fig.add_subplot(gs[1,1])


        N_most_recent = 1000

        self.plotXY(ax_xy)
        self.plotAng(ax_ang)
        self.plotAction(ax_action)

        self.plotRecentR(ax_R_recent)
        self.plotAvgR(ax_R_avg)
        self.plotRecentAvgR(ax_R_window_avg)

        plt.tight_layout()

        plt.savefig(fst.combineDirAndFile(self.local_final_dir, 'plot_{}_R.png'.format(self.latest_str)))
        if not kwargs.get('noshow', False):
            plt.show()

        plt.close('all')





    def plotXY(self, ax, N_most_recent = 1000):
        ax.set_xlabel('iteration')
        ax.set_ylabel('x, y')
        ax.plot(self.robot_hist[-N_most_recent:,0], self.robot_hist[-N_most_recent:, 2])
        ax.plot(self.robot_hist[-N_most_recent:,0], self.robot_hist[-N_most_recent:, 3])


    def plotAng(self, ax, N_most_recent = 1000):

        ax.set_xlabel('iteration')
        ax.set_ylabel('angle')
        ax.plot(self.robot_hist[-N_most_recent:,0], self.robot_hist[-N_most_recent:, 4])


    def plotAction(self, ax, N_most_recent = 1000):

        ax.set_xlabel('iteration')
        ax.set_ylabel('action')
        ax.plot(self.robot_hist[-N_most_recent:,0], self.robot_hist[-N_most_recent:, 6])


    def plotAllR(self, ax, N_most_recent = 1000):

        ax.set_xlabel('iteration')
        ax.set_ylabel('R')
        ax.plot(self.robot_hist[:,0], self.robot_hist[:, 5])


    def plotRecentR(self, ax, N_most_recent = 1000):

        ax.set_xlabel('iteration')
        ax.set_ylabel('R_last_{}'.format(N_most_recent))
        ax.plot(self.robot_hist[-N_most_recent:,0], self.robot_hist[-N_most_recent:, 5])
        ax.axhline(self.robot_hist[-N_most_recent:, 5].mean(), linestyle='dashed', color='darkred')


    def plotAvgR(self, ax, N_most_recent = 1000):
        N_start = 400

        ax.set_xlabel('iteration')
        ax.set_ylabel('R_avg')
        ax.plot(self.robot_hist[N_start:,0], np.cumsum(self.robot_hist[N_start:, 5])/(1 + self.robot_hist[N_start:,0]))


    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return((cumsum[N:] - cumsum[:-N]) / float(N))


    def plotRecentAvgR(self, ax, N_most_recent = 1000):
        N_start = 400
        N_window = 1000

        ax.set_xlabel('iteration')
        ax.set_ylabel('R_{}_avg'.format(N_window))

        window_avg = self.running_mean(self.robot_hist[N_start:, 5], N_window)

        ax.plot(list(range(len(window_avg))), window_avg)




    def plotQAndActions(self, ax_list_Q, ax_list_act, **kwargs):
        ##################################### Q values as a function of position

        NN_model_fname = kwargs.get('NN_model_fname', self.last_NN_model_file_fullpath)

        # Pass the ax_list's in order of 0-7 of the targets

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

        target_rad = N_disc/17

        # Ugh I think I actually have to create an Agent until I do it in a more separated way...

        if self.params_info.get('state_type', 'position') == 'position':
            agent_class = DummyRobot_5state
        else:
            agent_class = DummyRobot_6state

        ag = Agent(
        agent_class=agent_class,
        features='DQN',
        N_steps=5*10**4,
        N_hidden_layer_nodes = self.params_info.get('N_hidden_layer_nodes', 50),
        two_hidden_layers = self.params_info.get('two_hidden_layers', False),
        double_DQN = False,
        NL_fn = 'tanh',
        save_params_to_file = False,
        )

        ag.loadModel(fname=NN_model_fname)


        pos = np.array([[[x,y] for y in np.linspace(*arena_lims, N_disc)] for x in np.linspace(*arena_lims, N_disc)])
        ang = np.expand_dims(np.full((N_disc,N_disc), 0), axis=2)

        f = lambda x: self.posAngleToDistances(x, 0)
        ds = np.apply_along_axis(f, 2, pos)

        for target in range(N_targets):

            last_target_pos = target_positions[target]
            #print('last target pos:', last_target_pos)
            # This is creating an array of tensors that we can feed into the trained NN to see its
            # current estimate of the Q function, etc. It uses angle=0, and the last target, and then
            # plots Q for all the pos.

            targ_pos_x = np.expand_dims(np.full((N_disc,N_disc), last_target_pos[0]), axis=2)
            targ_pos_y = np.expand_dims(np.full((N_disc,N_disc), last_target_pos[1]), axis=2)

            if self.params_info.get('state_type', 'position') == 'position':
                states = np.concatenate((pos, ang, targ_pos_x, targ_pos_y), axis=2)
            else:
                states = np.concatenate((ds, ang, targ_pos_x, targ_pos_y), axis=2)

            states = torch.tensor(states, dtype=torch.float32)

            NN_output = ag.forwardPass(ag.policy_NN, states)

            max_Q = (torch.max(NN_output, dim=2)[0]).detach().numpy()
            best_actions = (torch.argmax(NN_output, dim=2)).detach().numpy()


            ######### Plot Q vals
            if kwargs.get('plot_Q', True):

                ax_Q = ax_list_Q[target]
                col_plot_Q = ax_Q.matshow(max_Q.T, cmap='Reds', origin='lower')

                ax_Q.set_xlabel('x')
                ax_Q.set_ylabel('y')
                ax_Q.set_xticks([])
                ax_Q.set_yticks([])
                ax_Q.set_xticklabels([])
                ax_Q.set_yticklabels([])

                # The thing is plotted in terms of the indices of max_Q_Q, not the actual x and y vals...
                target_circle = plt.Circle(
                ((last_target_pos[0] - arena_lims[0])*N_disc/wall_length, (last_target_pos[1] - arena_lims[0])*N_disc/wall_length),
                target_rad, color='black')

                ax_Q.add_artist(target_circle)

            ######## Plot best actions
            if kwargs.get('plot_act', True):

                ax_best_act = ax_list_act[target]
                cm = LinearSegmentedColormap.from_list('my_cm', ['tomato','dodgerblue','seagreen','orange'], N=4)
                col_plot_bestact = ax_best_act.matshow(best_actions.T, cmap=cm, origin='lower', vmin=0, vmax=3)

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

                divider = make_axes_locatable(ax_best_act)
                cb_ax = divider.append_axes("right", size="10%", pad=0.05)
                #plt.colorbar(im, cax=cax)


                fig = plt.gcf()
                col_bar = fig.colorbar(col_plot_bestact, cax=cb_ax, ticks=[0,1,2,3], boundaries=np.arange(-.5,4.5,1))
                col_bar.ax.set_yticklabels(['F','B','CCW','CW'])

                ax_best_act.add_artist(target_circle)






    def loadArenaInfo(self):
        self.wall_length = 1.25
        self.xlims = np.array([-self.wall_length/2, self.wall_length/2])
        self.ylims = np.array([-self.wall_length/2, self.wall_length/2])
        self.bottom_corner = np.array([self.xlims[0], self.ylims[0]])


    def cornerOriginToCenterOrigin(self, pos):

        # This changes it so if your coords are so that the origin is in the
        # bottom left hand corner, now it's in the middle of the arena.

        center_origin_pos = pos + self.bottom_corner
        return(center_origin_pos)

    def centerOriginToCornerOrigin(self, pos):

        # This changes it so if your coords are so that the origin is in the
        # bottom left hand corner, now it's in the middle of the arena.

        corner_origin_pos = pos - self.bottom_corner
        return(corner_origin_pos)


    def posAngleToCollideVec(self, pos, ang):
        #
        # This takes a proposed position and angle, and returns the vector
        # that would collide with the wall it would hit at that angle.
        #
        # This assumes *lower left* origin coords.
        #
        # The angle bds here are the 4 angle bounds that determine
        # for a given x,y which wall the ray in that direction would collide with.
        # The order is: top wall, left wall, bottom wall, right wall

        x, y = pos

        angle_bds = [
        np.arctan2(self.wall_length-y, self.wall_length-x),
        np.arctan2(self.wall_length-y, -x),
        np.arctan2(-y, -x),
        np.arctan2(-y, self.wall_length-x),
        ]

        ray_x, ray_y = 0, 0

        small_angle = 0.02

        if (ang >= angle_bds[0]) and (ang < angle_bds[1]):
            ray_y = self.wall_length - y
            if abs(ang) > small_angle :
                ray_x = ray_y/tan(ang) #, self.wall_length*np.sign(tan(ang)))
            else:
                if np.sign(tan(ang)) > 0:
                    ray_x = self.wall_length - x
                else:
                    ray_x = -x

        elif (ang >= angle_bds[1]) or (ang < angle_bds[2]):
            ray_x = -x
            ray_y = ray_x*tan(ang)

        elif (ang >= angle_bds[2]) and (ang < angle_bds[3]):
            ray_y = -y
            if abs(ang) > small_angle :
                ray_x = ray_y/tan(ang) #, self.wall_length*np.sign(tan(ang)))
            else:
                if np.sign(tan(ang)) > 0:
                    ray_x = self.wall_length - x
                else:
                    ray_x = -x

        elif (ang >= angle_bds[3]) and (ang < angle_bds[0]):
            ray_x = self.wall_length - x
            ray_y = ray_x*tan(ang)

        return(np.array([ray_x, ray_y]))



    def posAngleToDistances(self, pos, ang):

        pos = self.centerOriginToCornerOrigin(pos)

        d1 = np.linalg.norm(self.posAngleToCollideVec(pos, ang))
        d2 = np.linalg.norm(self.posAngleToCollideVec(pos, ang + pi/2))
        d3 = np.linalg.norm(self.posAngleToCollideVec(pos, ang - pi/2))

        return(np.array([d1, d2, d3]))



######################### Other misc fns

    def dirVarsSetup(self, run_dir):
        #
        # Just sets a few member vars from run_dir

        self.run_dir = run_dir

        self.remote_run_dir_fullpath = fst.combineDirAndFile('/home/pi/robot_car/misc_runs', self.run_dir)
        self.remote_resume_file = fst.combineDirAndFile(self.remote_run_dir_fullpath, 'resume.json')
        self.remote_params_file = fst.combineDirAndFile(self.remote_run_dir_fullpath, 'params.json')
        self.remote_robot_hist_dir = fst.combineDirAndFile(self.remote_run_dir_fullpath, 'robot_hist')
        self.remote_NN_dir = fst.combineDirAndFile(self.remote_run_dir_fullpath, 'NN_info')

        self.local_base_dir = 'misc_runs/'
        self.local_final_dir = fst.combineDirAndFile(self.local_base_dir, self.run_dir)
        self.robot_hist_dir = fst.combineDirAndFile(self.local_final_dir, 'robot_hist')
        self.NN_dir = fst.combineDirAndFile(self.local_final_dir, 'NN_info')

        self.resume_file = fst.combineDirAndFile(self.local_final_dir, 'resume.json')
        self.params_file = fst.combineDirAndFile(self.local_final_dir, 'params.json')


    def checkRunDirExists(self):

        assert hasattr(self, 'local_final_dir'), 'self.local_final_dir not defined! Run dirVarsSetup() first'

        return(os.path.exists(self.local_final_dir))


    def dirSetup(self):

        if not self.checkRunDirExists():
            print('\n\nDir {} doesnt exist, creating.'.format(self.local_final_dir))
            print('\ncreating dirs:')
            print(self.local_final_dir)
            print(self.robot_hist_dir)
            print(self.NN_dir)
            # Make dirs
            fst.makeDir(self.local_final_dir)
            fst.makeDir(self.robot_hist_dir)
            fst.makeDir(self.NN_dir)




################################## For doing it via CLI

if __name__ == '__main__':

    # arguments to be read in via CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--nofetch', action='store_true', default=False)
    parser.add_argument('--Qvid', action='store_true', default=False)
    parser.add_argument('--noshow', action='store_true', default=False)
    args = parser.parse_args()

    kw = vars(args)

    vt = VizTools()
    vt.loadArenaInfo()
    vt.plotProgress(args.path, **kw)

    if args.Qvid:
        vt.createQVid(args.path, **kw)





#
