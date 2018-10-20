import time
import RPi.GPIO as GPIO


class Compass:

    def __init__(self):

        self.SETTINGS_FILE = "RTIMULib"
        s = RTIMU.Settings(self.SETTINGS_FILE)
        self.imu = RTIMU.RTIMU(s)


        if (not self.imu.IMUInit()):
        	print('IMU init failed.')
        	sys.exit(1)

        self.imu.setSlerpPower(0.02)
        self.imu.setGyroEnable(True)
        self.imu.setAccelEnable(True)
        self.imu.setCompassEnable(True)

        self.poll_interval = self.imu.IMUGetPollInterval()
        print('poll interval: ', self.poll_interval)


    def getReading(self):

        try:
        	while True:

        		if self.imu.IMURead():
        			data = self.imu.getIMUData()
        			fusion_pose = data["fusionPose"]
        			'''Gyro = data["gyro"]
        			print('\ngyro: {:.4f}, {:.4f}, {:.4f}'.format(Gyro[0], Gyro[1], Gyro[2]))
        			print('fusion: {:.4f}, {:.4f}, {:.4f}'.format(fusion_pose[0], fusion_pose[1], fusion_pose[2]))'''
                    return(fusion_pose)

        except:
        	print('error in getting IMU reading')


    def readCompassLoop(self, test_time=10):

        start_time = time.time()

        try:
        	while True:

                if time.time()-start_time > test_time:
                    break
                    
                reading = self.getReading()
                print('fusion: {:.4f}, {:.4f}, {:.4f}'.format(reading[0], reading[1], reading[2]))
                time.sleep(self.poll_interval*1.0/1000.0)

        except KeyboardInterrupt:
            print('exit from readCompassLoop')


    def __del__(self):
        pass
