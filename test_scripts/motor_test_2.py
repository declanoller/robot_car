import sys
import time
import RPi.GPIO as GPIO

Forward=11
Backward=12
sleeptime=1

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(Forward, GPIO.OUT)
GPIO.setup(Backward, GPIO.OUT)

def forward(x):
    GPIO.output(Forward, GPIO.HIGH)
    print("Moving Forward")
    time.sleep(x)
    GPIO.output(Forward, GPIO.LOW)

def reverse(x):
    GPIO.output(Backward, GPIO.HIGH)
    print("Moving Backward")
    time.sleep(x)
    GPIO.output(Backward, GPIO.LOW)

while True:

    try:
        forward(1)
        reverse(1)
    except:
        GPIO.output(Forward, GPIO.LOW)
        GPIO.output(Backward, GPIO.LOW)
        break


GPIO.cleanup()
