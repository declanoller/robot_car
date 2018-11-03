import RPi.GPIO as GPIO
from Compass import Compass

compass = Compass()
print('Compass object created.')

reading = compass.getReading()
print('compass reading: ', reading)
