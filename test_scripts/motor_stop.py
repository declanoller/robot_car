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

GPIO.output(Forward, GPIO.LOW)
GPIO.output(Backward, GPIO.LOW)

GPIO.cleanup()
