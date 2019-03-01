import sys
sys.path.append('./classes')
from Robot import Robot


rob = Robot(save_hist=True, quiet=True, arena_mode=True, reward_method = 'software', compass_correction_file='18-02-2019_12-25-37_compass_cal.json')

rob.directControl()
