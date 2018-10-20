import sys
sys.path.append('.')
import RTIMU
import time
import math
import matplotlib.pyplot as plt
import numpy as np

SETTINGS_FILE = "RTIMULib"
s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)


if (not imu.IMUInit()):
	print('IMU init failed.')
	sys.exit(1)



imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

poll_interval = imu.IMUGetPollInterval()
print('poll interval: ', poll_interval)

gyro_meas = []
fusion_meas = []

try:
	while True:

		if imu.IMURead():
			data = imu.getIMUData()
			fusion_pose = data["fusionPose"]
			Gyro = data["gyro"]
			print('\ngyro: {:.4f}, {:.4f}, {:.4f}'.format(Gyro[0], Gyro[1], Gyro[2]))
			print('fusion: {:.4f}, {:.4f}, {:.4f}'.format(fusion_pose[0], fusion_pose[1], fusion_pose[2]))

			gyro_meas.append(Gyro)
			fusion_meas.append(fusion_pose)

			time.sleep(poll_interval*1.0/1000.0)

except:
	pass


gyro_meas = np.array(gyro_meas)
print(gyro_meas.shape)
fusion_meas = np.array(fusion_meas)
print(fusion_meas.shape)



fig, axes = plt.subplots(2, 3, figsize=(16,12))

for i in range(3):

	axes[0, i].plot(gyro_meas[:,i])
	axes[1, i].plot(fusion_meas[:,i])



plt.savefig('meas.png')






#
