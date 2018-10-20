import time
import RPi.GPIO as GPIO
from Motor import Motor
from Sonar import Sonar
from Compass import Compass


class Robot:

    def __init__(self, motor_enable=True, sonar_enable=True, compass_enable=True):


        #GPIO Mode (BOARD / BCM)
        GPIO.setmode(GPIO.BOARD)

        if motor_enable:
            motor = Motor(left_forward_pin=33, left_reverse_pin=31, right_forward_pin=37, right_reverse_pin=35)
            print('Motor object created.')
        if sonar_enable:
            sonar = Sonar(GPIO_TRIGGER=10, GPIO_ECHO=8)
            print('Sonar object created.')
        if compass_enable:
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            compass = Compass()
            print('Compass object created.')






    def testAllDevices(self):

        if motor_enable:
            print('testing motor!')
            motor.wheelTest(test_time=5)
        if sonar_enable:
            print('testing sonar!')
            sonar.distanceTestLoop(test_time=3)
        if compass_enable:
            print('testing compass!')
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            compass.readCompassLoop(test_time=4)






    def __del__(self):
        GPIO.cleanup()
