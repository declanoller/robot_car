import time
import RPi.GPIO as GPIO


class Sonar:

    def __init__(self, GPIO_TRIGGER=10, GPIO_ECHO=8):

        # From now on the distance is returned in units of METERS.

        GPIO.setmode(GPIO.BOARD)

        self.GPIO_TRIGGER = GPIO_TRIGGER
        self.GPIO_ECHO = GPIO_ECHO

        #set GPIO direction (IN / OUT)
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)



    def distance(self):
        # set Trigger to HIGH
        GPIO.output(self.GPIO_TRIGGER, GPIO.HIGH)

        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, GPIO.LOW)

        StartTime = time.time()
        StopTime = time.time()

        # save StartTime
        while GPIO.input(self.GPIO_ECHO) == 0:
            StartTime = time.time()

        # save time of arrival
        while GPIO.input(self.GPIO_ECHO) == 1:
            StopTime = time.time()

        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2

        return(distance/100.0)


    def distanceTestLoop(self, test_time=10):

        start_time = time.time()

        try:
            while True:

                if time.time()-start_time > test_time:
                    break

                dist = self.distance()
                print ("Measured Distance = %.1f cm" % dist)
                time.sleep(0.2)

            # Reset by pressing CTRL + C
        except KeyboardInterrupt:
            print("Measurement stopped by User")
            #GPIO.cleanup()

        print('done testing sonar.')



    def __del__(self):
        pass
