import RPi.GPIO as GPIO

IR_pin = 7


def setup():
	GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
	GPIO.setup(IR_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def loop():
	i = 0
	while True:
		if (0 == GPIO.input(IR_pin)):
			print("Detected Barrier!", i)
			i += 1


def destroy():
	GPIO.cleanup()                     # Release resource

if __name__ == '__main__':     # Program start from here
	setup()
	try:
		loop()
	except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
		destroy()
