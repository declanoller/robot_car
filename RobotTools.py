from Agent import Agent

from Robot import Robot
import numpy as np
import FileSystemTools as fst
from copy import copy
import json

'''
So this will be so you can either start a new episode with the robot,
or pick up exactly where you left off with a previous one (loading its
NN model, history, etc).

If you're starting a new one, just pass it the agent params.

If you're resuming one, pass to the arg resume_dir the dir you
want to pick up where you left off.

--self.run_params['resume_dir'] will be the dir for the whole run.
--self.run_params['robot_hist_dir'] will be where the saved histories for the chunks
get saved.
--self.run_params['NN_info_dir'] will be where the NN model/optimizer weights
get saved.



---It might make sense for Agent to save its own params, so it can then load
them and not have to deal with stuff like the fact that it changed epsilon since
starting, but the original run_params will have the original one...


Things to have in resume_file:
-next starting chunk, iter ✓
-last chunk, iteration completed ✓



Have dirs with:
-NN, optim. weights ✓
-robot hists ✓


make sure to save:
-NN weights (target and policy both) ✓
-experiences ✓
-optimizer ✓
-robot_hist ✓
-epsilon, exp_pos, etc ✓

-debug file // ehhhhh fuck it, I don't think I need these for now and I don't wanna
deal with moving them around.



have to reload:
-NN weights (target and policy both) ✓
-experiences ✓
-optimizer ✓
-epsilon etc ✓


-------remember, gotta make N_steps only the chunk size for Agent...

----------- gotta make thing to make Robot start at later iter.


------- add thing to stitch them together at the end

'''




class RobotTools:

    def __init__(self, **kwargs):

        self.run_params = {}
        self.run_params['resume_dir'] = kwargs.get('resume_dir', None)

        # This will make it so Agent.py doesn't save its params to file each run.
        self.params_file_basename = 'params.json'
        self.resume_file_basename = 'resume.json'



        if self.run_params['resume_dir'] is None:
            # Starting a new run

            # Run dir stuff
            self.run_params['fname_notes'] = kwargs.get('fname_notes', '')
            self.run_params['datetime_str'] = fst.getDateString()
            self.run_params['base_dir'] = kwargs.get('base_dir', 'misc_runs')
            self.run_params['save_params_to_file'] = False
            self.run_params['resume_dir'] = fst.combineDirAndFile(self.run_params['base_dir'], 'Robot_{}_{}'.format(self.run_params['fname_notes'], self.run_params['datetime_str']))

            self.run_params['save_params_to_file'] = False
            self.run_params['debug_enable'] = False

            self.createStartingDirs()

            self.setUpTimingChunks(**kwargs)

            # This combines both dicts, so the agent should get everything that was passed
            # to kwargs as well as stuff created by the stuff above.
            self.run_params = {**kwargs, **self.run_params}

            # Save the run params
            self.saveParameters()

            self.resume_info = {
            'next_starting_chunk' : 0,
            'next_starting_iteration' : 0,
            'last_chunk_completed' : -1,
            'last_iteration_completed' : -1
            }

            # Save the resume_info
            self.saveResumeFile()



        # So either way (whether you're resuming or starting anew), you can proceed from here.
        # Either it created the dir/relevant files, or they already exist and you can load them.

        # Reload run parameters
        self.loadParameters()

        # Reload resume info
        self.loadResumeFile()

        print('\n\nParameters: \n')
        fst.dictPrettyPrint(self.run_params)

        print('\n\nResume info: \n')
        fst.dictPrettyPrint(self.resume_info)

        print('Creating agent...\n')
        self.agent = Agent(agent_class=Robot, **self.run_params)





    def run(self):

        start_chunk = self.resume_info['next_starting_chunk']
        end_chunk = self.run_params['N_chunks']

        # So here, every iteration, I'll make it reload the Agent params,
        # NN, and optimizer. At the end of each episode, it should save them.

        for current_chunk in range(start_chunk, end_chunk):

            print('\n\n\n<---------------- Starting current chunk {} --------------->\n\n'.format(current_chunk))

            if current_chunk > 0:
                self.loadAllAgentInfo()

            ret = self.agent.DQNepisode()

            if ret['exit_code'] == 1:
                print('Error in DQNepisode(), exit_code returned 1. Breaking now, fix the problem.')
                break

            print('\nCurrent chunk {} finished. Saving info and data now...\n'.format(current_chunk))
            # This will use the current chunk/iter number, so call it before updating those.
            self.saveAllAgentInfo()
            self.saveRobotHist()

            # At this point, we can assume that it has completed that chunk, and
            # update resume.json.
            self.incrementResumeInfo()
            self.saveResumeFile()






######################## Others


    def saveAllAgentInfo(self):

        save_pair_list = [
        (self.agent.saveModel, self.run_params['NN_model_fname']),
        (self.agent.saveOptimizer, self.run_params['NN_optim_fname']),
        (self.agent.saveParams, self.run_params['NN_params_fname']),
        (self.agent.saveExperiences, self.run_params['NN_exp_fname']),
        ]

        # Will make it go from iter 0-999, for ex.
        for fn, fname_template in save_pair_list:
            fname = fname_template.format(
            self.resume_info['next_starting_chunk'],
            self.resume_info['next_starting_iteration'],
            self.resume_info['last_iteration_completed'] + self.run_params['N_iterations_per_chunk'])
            fn(fname=fname)





    def loadAllAgentInfo(self):

        save_pair_list = [
        (self.agent.loadModel, self.run_params['NN_model_fname']),
        (self.agent.loadOptimizer, self.run_params['NN_optim_fname']),
        (self.agent.loadParams, self.run_params['NN_params_fname']),
        (self.agent.loadExperiences, self.run_params['NN_exp_fname']),
        ]

        for fn, fname_template in save_pair_list:
            fname = fname_template.format(
            self.resume_info['next_starting_chunk'] - 1,
            self.resume_info['next_starting_iteration'] - self.run_params['N_iterations_per_chunk'],
            self.resume_info['last_iteration_completed'])
            fn(fname=fname)

        ### Might need to make a separate fn for this later because it's more
        # interfacting with Robot than Agent, but I'll stick it here for now.
        self.agent.agent.initial_iteration = self.resume_info['next_starting_iteration']


    def saveRobotHist(self):

        # Includes the robot_hist dir.
        fname = self.run_params['robot_hist_fname'].format(
        self.resume_info['next_starting_chunk'],
        self.resume_info['next_starting_iteration'],
        self.resume_info['last_iteration_completed'] + self.run_params['N_iterations_per_chunk'])

        self.agent.agent.saveHist(fname=fname)




    def saveResumeFile(self):
        # Save the resume_info
        self.resume_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.resume_file_basename)
        with open(self.resume_file, 'w+') as f:
            json.dump(self.resume_info, f, indent=4)


    def loadResumeFile(self):
        self.resume_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.resume_file_basename)
        with open(self.resume_file, 'r') as f:
            self.resume_info = json.load(f)


    def saveParameters(self):
        self.params_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.params_file_basename)
        with open(self.params_file, 'w+') as f:
            json.dump(self.run_params, f, indent=4)


    def loadParameters(self):
        self.params_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.params_file_basename)
        with open(self.params_file, 'r') as f:
            self.run_params = json.load(f)



    def createStartingDirs(self):

        print('\n\nCreating new run directory...\n')

        # Creates:
        # --resume_dir
        # --NN_info_dir
        # --robot_hist_dir
        # --debug_files_dir

        fst.makeDir(self.run_params['resume_dir'])
        print('run dir: ', self.run_params['resume_dir'])

        self.params_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.params_file_basename)
        print('params file:', self.params_file)
        self.resume_file = fst.combineDirAndFile(self.run_params['resume_dir'], self.resume_file_basename)
        print('resume file:', self.resume_file)

        # Create other dirs
        self.run_params['NN_info_dir'] = fst.combineDirAndFile(self.run_params['resume_dir'], 'NN_info')
        fst.makeDir(self.run_params['NN_info_dir'])
        print('NN and optim. weights dir:', self.run_params['NN_info_dir'])

        self.run_params['NN_model_fname'] = fst.combineDirAndFile(self.run_params['NN_info_dir'], 'NN_model_chunk_{}_iterations_{}-{}.txt')
        self.run_params['NN_params_fname'] = fst.combineDirAndFile(self.run_params['NN_info_dir'], 'NN_params_chunk_{}_iterations_{}-{}.txt')
        self.run_params['NN_optim_fname'] = fst.combineDirAndFile(self.run_params['NN_info_dir'], 'NN_optim_chunk_{}_iterations_{}-{}.txt')
        self.run_params['NN_exp_fname'] = fst.combineDirAndFile(self.run_params['NN_info_dir'], 'NN_exp_chunk_{}_iterations_{}-{}.txt')



        self.run_params['robot_hist_dir'] = fst.combineDirAndFile(self.run_params['resume_dir'], 'robot_hist')
        fst.makeDir(self.run_params['robot_hist_dir'])
        print('Robot history dir:', self.run_params['robot_hist_dir'])

        self.run_params['robot_hist_fname'] = fst.combineDirAndFile(self.run_params['robot_hist_dir'], 'robot_hist_chunk_{}_iterations_{}-{}.txt')

        '''self.run_params['debug_files_dir'] = fst.combineDirAndFile(self.run_params['resume_dir'], 'debug_files')
        fst.makeDir(self.run_params['debug_files_dir'])
        print('Debug files dir:', self.run_params['debug_files_dir'])'''


        print('\n\n')


    def setUpTimingChunks(self, **kwargs):

        # Stuff for timing
        self.run_params['N_total_iterations'] = kwargs.get('N_total_iterations', None)
        assert self.run_params['N_total_iterations'] is not None, 'Need to provide N_total_iterations'
        # This will cause it to round down, so if you ask for 1001 total iterations with
        # 50 chunks, it will only do 20 per chunk.
        self.run_params['N_chunks'] = kwargs.get('N_chunks', 50)
        self.run_params['N_iterations_per_chunk'] = kwargs.get('N_iterations_per_chunk', int(self.run_params['N_total_iterations']/self.run_params['N_chunks']))

        self.run_params['N_steps'] = self.run_params['N_iterations_per_chunk']


    def incrementResumeInfo(self):

        self.resume_info['next_starting_chunk'] += 1
        self.resume_info['next_starting_iteration'] += self.run_params['N_iterations_per_chunk']
        self.resume_info['last_iteration_completed'] += self.run_params['N_iterations_per_chunk']

        # We start at 0, so you've only actually completed chunk 0 at the end of it.
        if self.resume_info['last_chunk_completed'] != 0:
            self.resume_info['last_chunk_completed'] += 1






















#
