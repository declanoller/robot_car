import time
import RPi.GPIO as GPIO

left_forward_pin = 33
left_reverse_pin = 31
right_forward_pin = 37
right_reverse_pin = 35
sleeptime=1

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(left_forward_pin, GPIO.OUT)
GPIO.setup(left_reverse_pin, GPIO.OUT)
GPIO.setup(right_forward_pin, GPIO.OUT)
GPIO.setup(right_reverse_pin, GPIO.OUT)

def leftWheelForward(x):
    GPIO.output(left_forward_pin, GPIO.HIGH)
    print("Moving L Forward")
    time.sleep(x)
    GPIO.output(left_forward_pin, GPIO.LOW)

def leftWheelReverse(x):
    GPIO.output(left_reverse_pin, GPIO.HIGH)
    print("Moving L Backward")
    time.sleep(x)
    GPIO.output(left_reverse_pin, GPIO.LOW)



def rightWheelForward(x):
    GPIO.output(right_forward_pin, GPIO.HIGH)
    print("Moving R Forward")
    time.sleep(x)
    GPIO.output(right_forward_pin, GPIO.LOW)

def rightWheelReverse(x):
    GPIO.output(right_reverse_pin, GPIO.HIGH)
    print("Moving R Backward")
    time.sleep(x)
    GPIO.output(right_reverse_pin, GPIO.LOW)





def stopAllWheels():
    GPIO.output(left_forward_pin, GPIO.LOW)
    GPIO.output(left_reverse_pin, GPIO.LOW)
    GPIO.output(right_forward_pin, GPIO.LOW)
    GPIO.output(right_reverse_pin, GPIO.LOW)

while True:

    try:
        leftWheelForward(1)
        leftWheelReverse(1)
        rightWheelForward(1)
        rightWheelReverse(1)
    except:
        stopAllWheels()
        break


GPIO.cleanup()
