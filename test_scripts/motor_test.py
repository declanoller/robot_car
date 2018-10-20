from gpiozero import Motor
from time import sleep


motor = Motor(17, 18)

while True:
    motor.forward()
    sleep(1)
    motor.stop()
    motor.backward()
    sleep(1)
    motor.stop()
