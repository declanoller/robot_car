from Robot import Robot
#from Agent import Agent



rob = Robot(save_hist=True, quiet=True, arena_mode=True, compass_correction_file='13-02-2019_12-41-43_compass_cal.json')

rob.directControl()

#rob.testAllDevices()
