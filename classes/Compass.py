import sys
sys.path.append('.')
import RTIMU
import time
import matplotlib.pyplot as plt
import numpy as np
import FileSystemTools as fst
from math import pi, sin, copysign, floor, ceil
from scipy.interpolate import interp1d
import json

class Compass:

	def __init__(self, compass_correction=None, compass_correction_file=None, pi_max=True, raw_reading=False, flip_x=True):

		self.SETTINGS_FILE = 'RTIMULib'
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

		# This will make the angle between -pi and pi, which we usually want.
		# However, for debugging, you might want the simplest thing.
		self.restrict_to_pi_max = pi_max
		self.flip_x_direction = flip_x

		self.compass_correction = compass_correction
		# Pass compass_correction as a dict with "meas" being the ones you measured
		# and "ideal" being the ideal ones, and it will take care of the rest. They both
		# have to be np arrays.

		if compass_correction_file is not None:
			with open(compass_correction_file, 'r') as f:
				self.compass_correction = json.load(f)
				self.compass_correction['ideal_angles'] = np.array(self.compass_correction['ideal_angles'])
				self.compass_correction['meas_angles'] = np.array(self.compass_correction['meas_angles'])



		if raw_reading:
			self.restrict_to_pi_max = False
			self.compass_correction = None
			self.flip_x_direction = False

		if self.compass_correction is not None:
			print('applying compass correction from supplied data.')
			self.correction_interp = self.getCorrectionInterpolation(self.compass_correction)

		# I think I need to do these to keep emptying the FIFO compass buffer in parallel.
		self.last_reading = None



	def getReading(self):

		try:

			while True:

				if self.imu.IMURead():
					data = self.imu.getIMUData()

					fusion_pose = np.array(data['fusionPose'])


					if self.flip_x_direction:
						fusion_pose[2] = pi - fusion_pose[2]


					if self.restrict_to_pi_max:
						plane_switch = 0.5*(1 - copysign(1, sin(fusion_pose[2])))
						fusion_pose[2] = fusion_pose[2]%(2*pi) - 2*pi*plane_switch


					if self.compass_correction is not None:
						# Right, so the angle from the compass is actually between
						# -pi and pi by default, so the one coming out of the correction
						# will be also, but I have to do the pi - angle thing for now because
						# the compass is flipped upside down, so it rotates in the wrong direction,
						# and the pi is there because it's 180 degrees off from where I'd like
						# it to point for angle=0. So that makes the angle end up between
						# 0 and 2pi.
						try:
							if fusion_pose[2] < min(self.compass_correction['meas_angles']):
								fusion_pose[2] = min(self.compass_correction['meas_angles'])
							if fusion_pose[2] > max(self.compass_correction['meas_angles']):
								fusion_pose[2] = max(self.compass_correction['meas_angles'])

							fusion_pose[2] += self.correction_interp(fusion_pose[2])


							if self.restrict_to_pi_max:
								plane_switch = 0.5*(1 - copysign(1, sin(fusion_pose[2])))
								fusion_pose[2] = fusion_pose[2]%(2*pi) - 2*pi*plane_switch

						except:
							crash_str = 'problem in using interp. Reading passed to interp is: {:.3f}'.format(fusion_pose[2])
							f = open('crash_file.txt', 'w+')
							f.write(crash_str)
							f.close()
							print(crash_str)
							return(0)


					self.last_reading = fusion_pose[2]
					time.sleep(self.poll_interval*1.0/1000.0)
					return(fusion_pose)

		except KeyboardInterrupt:
			print('error in getting IMU reading')


	def getCompassDirection(self):
		#[2] is the one for the plane parallel with the ground.
		#return(self.getReading()[2])
		# I think I shouldn't do getReading() here, because the assumption is
		# that self.last_reading is constantly getting reset to be the last one
		# by the loop thread.
		#self.getReading()
		return(self.last_reading)



	def readCompassLoop(self, **kwargs):


		test_time = kwargs.get('test_time', 10)
		save_plot = kwargs.get('save_plot', False)
		save_dat = kwargs.get('save_dat', False)
		print_readings = kwargs.get('print_readings', False)

		start_time = time.time()
		fusion_meas = []

		while True:

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




	def getCorrectionInterpolation(self, compass_correction):


		ideal_angles = compass_correction['ideal_angles']
		meas_angles = compass_correction['meas_angles']

		min_meas = compass_correction['min_raw'] - 0.05*abs(compass_correction['min_raw'])
		max_meas = compass_correction['max_raw'] + 0.05*abs(compass_correction['max_raw'])


		matched = list(zip(meas_angles, ideal_angles))

		matched.append((min_meas, min_meas))
		matched.append((max_meas, max_meas))

		matched_sorted = sorted(matched, key=lambda x: x[0]) # Sort by increasing meas. angles

		meas_angles = np.array([x[0] for x in matched_sorted])
		ideal_angles = np.array([x[1] for x in matched_sorted])

		diff = ideal_angles - meas_angles
		diff_min = diff[0]
		diff_max = diff[1]

		for i in range(len(diff)):
			if diff[i] < -pi:
				diff[i] += 2*pi
			if diff[i] > pi:
				diff[i] -= 2*pi

		interp_diff = interp1d(meas_angles, diff, kind='linear')
		return(interp_diff)





#
