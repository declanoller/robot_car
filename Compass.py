import sys
sys.path.append('.')
import RTIMU
import time
import matplotlib.pyplot as plt
import numpy as np
import FileSystemTools as fst

class Compass:

	def __init__(self):

		self.SETTINGS_FILE = 'CALIBRATED_RTIMULib'
		self.s = RTIMU.Settings(self.SETTINGS_FILE)
		self.imu = RTIMU.RTIMU(self.s)


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
					data = self.imu.getIMUData()

					fusion_pose = data['fusionPose']

					time.sleep(self.poll_interval*1.0/1000.0)
					return(fusion_pose)

		except KeyboardInterrupt:
			print('error in getting IMU reading')


	def getCompassDirection(self):
		#[2] is the one for the plane parallel with the ground.
		return(self.getReading()[2])

	def readCompassLoop(self, test_time=10, save_plot=False):

		start_time = time.time()
		fusion_meas = []

		while True:

				if time.time()-start_time > test_time:
					break

				try:
					reading = None
					reading = self.getReading()
					if reading is None:
						break
					fusion_meas.append(reading)
					print('fusion: {:.4f}, {:.4f}, {:.4f}'.format(reading[0], reading[1], reading[2]))
					time.sleep(self.poll_interval*1.0/1000.0)
				except KeyboardInterrupt:
					print('interrupted in read loop')
					break


		fusion_meas = np.array(fusion_meas)
		print(fusion_meas.shape)

		fig, axes = plt.subplots(1, 3, figsize=(16,6))
		for i in range(3):
			axes[i].plot(fusion_meas[:,i])

		fname = 'compass_meas_{}'.format(fst.getDateString())
		np.savetxt(fname+'.dat', fusion_meas)
		plt.savefig(fname+'.png')


		print('done testing compass.')
