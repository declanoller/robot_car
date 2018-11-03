import time
import RPi.GPIO as GPIO
import sys
sys.path.append('.')
import RTIMU

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

		print('IMU init successful.')
		self.poll_interval = self.imu.IMUGetPollInterval()
		print('poll interval: ', self.poll_interval)


	def getReading(self):

		try:
			while True:
				
				if self.imu.IMURead():
					print('read IMU')
					data = self.imu.getIMUData()
					fusion_pose = data['fusionPose']

					time.sleep(self.poll_interval*1.0/1000.0)
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

		print('done testing compass.')
