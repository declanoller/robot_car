from Robot import Robot
#from Agent import Agent



rob = Robot(motor_enable=True, sonar_enable=True, compass_enable=True, MQTT_enable=True)

rob.directControl()

#rob.testAllDevices()
