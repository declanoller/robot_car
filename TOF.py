import time
import RPi.GPIO as GPIO
import VL53L0X
from scipy.interpolate import interp1d
import numpy as np

class TOF:

    def __init__(self, GPIO_SHUTDOWN, i2c_address):

        # From now on the distance is returned in units of METERS.
        print('\n\n')

        GPIO.setmode(GPIO.BOARD)
        self.GPIO_SHUTDOWN = GPIO_SHUTDOWN

        #set GPIO direction (IN / OUT)
        GPIO.setup(self.GPIO_SHUTDOWN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(self.GPIO_SHUTDOWN, GPIO.LOW)
        time.sleep(0.5)
        self.tof = VL53L0X.VL53L0X(i2c_address=0x29)
        time.sleep(0.1)
        GPIO.output(self.GPIO_SHUTDOWN, GPIO.HIGH)
        time.sleep(0.1)
        self.tof.change_address(i2c_address)
        #time.sleep(0.1)
        #self.tof.open()
        #time.sleep(0.1)
        #self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
        self.timing = max(self.tof.get_timing(), 20000)/1000000.00

        ideal_dist = np.array([-10.0, 0.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 10000])/100.0
        meas_dist = np.array([-10.0, 0.0, 80.0, 87.0, 97.0, 105.0, 112.0, 120.0, 130.0, 10000])/100.0
        self.dist_correct = interp1d(meas_dist, ideal_dist, kind='linear')
        #print('distance right after creation: ', self.distance())


    def tofOpen(self):
        time.sleep(0.1)
        self.tof.open()

    def tofStartRanging(self):
        time.sleep(0.1)
        self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)


    def distance(self):
        # get_distance returns it in mm.
        time.sleep(self.timing)
        mm_dist = self.tof.get_distance()/1000.0
        return(self.dist_correct(mm_dist))


    def distanceTestLoop(self, test_time=10):

        start_time = time.time()

        try:
            while True:

                if time.time()-start_time > test_time:
                    break

                dist = self.distance()
                print ("Measured Distance = {:.5f} m".format(dist))
                time.sleep(self.timing)

            # Reset by pressing CTRL + C
        except KeyboardInterrupt:
            print("Measurement stopped by User")

        print('done testing TOF.')



    def __del__(self):
        #GPIO.setmode(GPIO.BOARD)
        self.tof.stop_ranging()
        print('Setting shutdown pin {} to low in TOF del.'.format(self.GPIO_SHUTDOWN))
        GPIO.output(self.GPIO_SHUTDOWN, GPIO.LOW)
        self.tof.close()
