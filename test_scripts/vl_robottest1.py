from Robot import Robot
#from Agent import Agent



rob = Robot(MQTT_enable=False)

rob.testAllDevices(test_duration = 6)
