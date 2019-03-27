import sys
sys.path.append('./classes')
from Agent import Agent
from Robot import Robot
import numpy as np
import FileSystemTools as fst
from RobotTools import RobotTools
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=None)
parser.add_argument('--new', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()

resume_file = '.resume_learn.json'

if not args.new:

    if args.resume:
        # If resuming, just get the last dir that ran from the file
        with open(resume_file) as f:
            resume_dict = json.load(f)
        resume_dir = resume_dict['dir']
    else:
        # If not, this assumes you're explicitly giving the path
        assert args.path is not None, 'Need to provide a path if not creating a new run or resuming from .resume_learn.txt'
        resume_dir = fst.combineDirAndFile('./misc_runs/', args.path)

    assert os.path.exists(resume_dir), 'resume dir {} does not exist!'.format(resume_dir)

    resume_dict['dir'] = resume_dir
    with open(resume_file, 'w+') as f:
        json.dump(resume_dict, f, indent=4)

    rt = RobotTools(resume_dir=resume_dir)
    rt.run()


else:
    rt = RobotTools(
    N_total_iterations = 2*10**5,
    N_chunks = 200,

    features='DQN',
    epsilon = 0.97,
    eps_decay = 0.9997,
    eps_min = 0.05,
    N_batch = 100,
    N_hidden_layer_nodes = 40,
    target_update = 500,
    double_DQN = False,
    NL_fn = 'tanh',
    two_hidden_layers = True,

    reward_method = 'software',
    debug_enable = 0,
    compass_correction_file = '18-02-2019_12-25-37_compass_cal.json',
    state_type = 'position'
    )

    resume_dict = {}
    resume_dict['dir'] = rt.getResumeDir()
    with open(resume_file, 'w+') as f:
        json.dump(resume_dict, f, indent=4)


    rt.run()





exit()















#
