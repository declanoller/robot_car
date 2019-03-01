import sys
sys.path.append('./classes')
from Robot import Robot
#from Agent import Agent



rob = Robot(arena_mode=False, motor_enable=True, sonar_enable=False, TOF_enable=False, MQTT_enable=False, compass_enable=False, quiet=True)

rob.directControl()

#rob.testAllDevices()
