import sys
sys.path.append('.')
import RTIMU
import time
import matplotlib.pyplot as plt
import numpy as np
import FileSystemTools as fst
from math import pi
from scipy.interpolate import interp1d

class Compass:

	def __init__(self, compass_correction=None, close_event=None):

		self.SETTINGS_FILE = 'BASEMENT_CAL_RTIMULib'
		self.s = RTIMU.Settings(self.SETTINGS_FILE)
		self.imu = RTIMU.RTIMU(self.s)


		if (not self.imu.IMUInit()):
			print('IMU init failed.')
			sys.exit(1)

		self.imu.setSlerpPower(0.02)
		self.imu.setGyroEnable(True)
		self.imu.setAccelEnable(True)
		self.imu.setCompassEnable(True)

		print('\nIMU init successful.')
		self.poll_interval = self.imu.IMUGetPollInterval()
		print('poll interval: ', self.poll_interval)

		self.compass_correction = compass_correction
		# Pass compass_correction as a dict with "meas" being the ones you measured
		# and "ideal" being the ideal ones, and it will take care of the rest. They both
		# have to be np arrays.

		if self.compass_correction is not None:
			print('applying compass correction from supplied data.')
			diff = self.compass_correction['ideal_angles'] - self.compass_correction['meas_angles']
			self.correction_interp = interp1d(self.compass_correction['meas_angles'], diff, kind='cubic')

		# I think I need to do these to keep emptying the FIFO compass buffer in parallel.
		self.last_reading = None
		self.close_event = close_event



	def getReading(self):

		try:

			while True:

				if self.imu.IMURead():
					data = self.imu.getIMUData()

					fusion_pose = np.array(data['fusionPose'])

					if self.compass_correction is not None:
						fusion_pose[2] += self.correction_interp(fusion_pose[2])
						fusion_pose[2] = pi - fusion_pose[2]

					self.last_reading = fusion_pose[2]
					time.sleep(self.poll_interval*1.0/1000.0)
					return(fusion_pose)

		except KeyboardInterrupt:
			print('error in getting IMU reading')


	def getCompassDirection(self):
		#[2] is the one for the plane parallel with the ground.
		#return(self.getReading()[2])
		return(self.last_reading)

	def readCompassLoop(self, test_time=10, save_plot=False, save_dat=False, print_readings=False):

		start_time = time.time()
		fusion_meas = []

		while True:

			if self.close_event is not None:
				if self.close_event.is_set():
					break

			if test_time != 'forever':
				if time.time()-start_time > test_time:
					break

			try:
				reading = None
				reading = self.getReading()
				if reading is None:
					break

				if save_dat or save_plot:
					fusion_meas.append(reading)

				if print_readings:
					print('fusion: {:.4f}, {:.4f}, {:.4f}'.format(reading[0], reading[1], reading[2]))

				time.sleep(self.poll_interval*1.0/1000.0)

			except:
				print('interrupted in read loop')
				break


		if save_dat or save_plot:
			fusion_meas = np.array(fusion_meas)
			print(fusion_meas.shape)
			fname = 'compass_meas_{}'.format(fst.getDateString())
			np.savetxt(fname+'.dat', fusion_meas)

		if save_plot:

			fig, axes = plt.subplots(1, 3, figsize=(16,6))
			for i in range(3):
				axes[i].plot(fusion_meas[:,i])
			plt.savefig(fname+'.png')

		print('done testing compass.')




#
