import sys
import time
import RPi.GPIO as GPIO

left_forward_pin=33
left_reverse_pin=32
right_forward_pin=37
right_reverse_pin=35

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(left_forward_pin, GPIO.OUT)
GPIO.setup(left_reverse_pin, GPIO.OUT)
GPIO.setup(right_forward_pin, GPIO.OUT)
GPIO.setup(right_reverse_pin, GPIO.OUT)

GPIO.output(left_forward_pin, GPIO.LOW)
GPIO.output(left_reverse_pin, GPIO.LOW)
GPIO.output(right_forward_pin, GPIO.LOW)
GPIO.output(right_reverse_pin, GPIO.LOW)


GPIO.cleanup()
