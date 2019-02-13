import time
import RPi.GPIO as GPIO


class Motor:

    def __init__(self, left_forward_pin=32, left_reverse_pin=33, right_forward_pin=37, right_reverse_pin=35):

        GPIO.setwarnings(False)
        print('GPIO getmode() in Motor before set: ', GPIO.getmode())
        GPIO.setmode(GPIO.BOARD)

        self.left_forward_pin = left_forward_pin
        self.left_reverse_pin = left_reverse_pin
        self.right_forward_pin = right_forward_pin
        self.right_reverse_pin = right_reverse_pin

        # A higher number here makes it turn for LONGER.
        carpet = 0.3
        concrete = 0.2
        concrete_ball_wheel = 0.7
        self.friction_fudge_factor = concrete_ball_wheel
        self.turn_time = 1.0
        self.straight_travel_time = 0.8
        self.tiny_turn_factor = 0.15

        self.motor_asymmetry_fudge_factor = 0.3

        GPIO.setup(self.left_forward_pin, GPIO.OUT)
        GPIO.setup(self.left_reverse_pin, GPIO.OUT)
        GPIO.setup(self.right_forward_pin, GPIO.OUT)
        GPIO.setup(self.right_reverse_pin, GPIO.OUT)

        self.stopAllWheels()


    def __del__(self):
        print('Motor __del__: setting pins low.')
        GPIO.setmode(GPIO.BOARD)
        print('GPIO mode in motor del:',GPIO.getmode())
        GPIO.setup(self.left_forward_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.left_reverse_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.right_forward_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.right_reverse_pin, GPIO.OUT, initial=GPIO.LOW)

        GPIO.output(self.left_forward_pin, GPIO.LOW)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)
        GPIO.output(self.right_forward_pin, GPIO.LOW)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)


    def goStraight(self):
        GPIO.output(self.left_forward_pin, GPIO.HIGH)
        GPIO.output(self.right_forward_pin, GPIO.HIGH)
        time.sleep(self.straight_travel_time*self.friction_fudge_factor)
        GPIO.output(self.left_forward_pin, GPIO.LOW)
        time.sleep(self.straight_travel_time*self.friction_fudge_factor*self.motor_asymmetry_fudge_factor)
        GPIO.output(self.right_forward_pin, GPIO.LOW)


    def goBackward(self):
        GPIO.output(self.left_reverse_pin, GPIO.HIGH)
        GPIO.output(self.right_reverse_pin, GPIO.HIGH)
        time.sleep(self.straight_travel_time*self.friction_fudge_factor)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)



    def turn90CCW(self):

        GPIO.output(self.left_forward_pin, GPIO.HIGH)
        GPIO.output(self.right_reverse_pin, GPIO.HIGH)
        time.sleep(self.turn_time*self.friction_fudge_factor*(1 + self.motor_asymmetry_fudge_factor))
        #time.sleep(self.turn_time*self.friction_fudge_factor)
        GPIO.output(self.left_forward_pin, GPIO.LOW)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)



    def turn90CW(self):

        GPIO.output(self.right_forward_pin, GPIO.HIGH)
        GPIO.output(self.left_reverse_pin, GPIO.HIGH)
        time.sleep(self.turn_time*self.friction_fudge_factor)
        GPIO.output(self.right_forward_pin, GPIO.LOW)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)



    def tinyCCW(self):

        # 0.3 on concrete does about 90, so let's do 0.1.

        GPIO.output(self.left_forward_pin, GPIO.HIGH)
        GPIO.output(self.right_reverse_pin, GPIO.HIGH)
        time.sleep(self.turn_time*self.tiny_turn_factor)
        GPIO.output(self.left_forward_pin, GPIO.LOW)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)



    def tinyCW(self):

        GPIO.output(self.right_forward_pin, GPIO.HIGH)
        GPIO.output(self.left_reverse_pin, GPIO.HIGH)
        time.sleep(self.turn_time*self.tiny_turn_factor)
        GPIO.output(self.right_forward_pin, GPIO.LOW)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)





    def leftWheelForward(self, x):
        GPIO.output(self.left_forward_pin, GPIO.HIGH)
        print("Moving L Forward")
        time.sleep(x)
        GPIO.output(self.left_forward_pin, GPIO.LOW)


    def leftWheelReverse(self, x):
        GPIO.output(self.left_reverse_pin, GPIO.HIGH)
        print("Moving L Backward")
        time.sleep(x)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)


    def rightWheelForward(self, x):
        GPIO.output(self.right_forward_pin, GPIO.HIGH)
        print("Moving R Forward")
        time.sleep(x)
        GPIO.output(self.right_forward_pin, GPIO.LOW)


    def rightWheelReverse(self, x):
        GPIO.output(self.right_reverse_pin, GPIO.HIGH)
        print("Moving R Backward")
        time.sleep(x)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)


    def stopAllWheels(self):
        GPIO.output(self.left_forward_pin, GPIO.LOW)
        GPIO.output(self.left_reverse_pin, GPIO.LOW)
        GPIO.output(self.right_forward_pin, GPIO.LOW)
        GPIO.output(self.right_reverse_pin, GPIO.LOW)


    def wheelTest(self, test_time=10):

        start_time = time.time()

        while True:

            if time.time()-start_time > test_time:
                break

            try:
                self.leftWheelForward(self.turn_time*self.friction_fudge_factor)
                self.leftWheelReverse(self.turn_time*self.friction_fudge_factor)
                self.rightWheelForward(self.turn_time*self.friction_fudge_factor)
                self.rightWheelReverse(self.turn_time*self.friction_fudge_factor)
            except:
                self.stopAllWheels()
                break





















#
