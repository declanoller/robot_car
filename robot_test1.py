from Robot import Robot
from Agent import Agent



rob = Robot(motor_enable=True, sonar_enable=False, compass_enable=False)

rob.directControl()

#rob.testAllDevices()
