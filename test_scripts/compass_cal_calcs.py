import matplotlib.pyplot as plt
import numpy as np
import FileSystemTools as fst
from math import pi
from scipy.interpolate import interp1d



def getCorrectionInterpolation(compass_correction):


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
	angles = np.linspace(min_meas, max_meas, 50)

	fig, axes = plt.subplots(1, 3, figsize=(16, 6))

	axes[0].set_xlabel('meas. angles')
	axes[0].set_ylabel('ideal. angles')
	axes[0].set_xlim(1.1*min_meas, 1.1*max_meas)
	axes[0].plot(meas_angles, ideal_angles, 'bo-', label='meas')
	axes[0].legend()


	axes[1].set_xlabel('meas. angles')
	axes[1].set_ylabel('diff. angles')
	axes[1].set_xlim(1.1*min_meas, 1.1*max_meas)
	axes[1].plot(meas_angles, diff, 'bo-', label='ideal - meas')
	axes[1].plot(angles, interp_diff(angles), 'r--', label='interp')
	axes[1].legend()


	axes[2].set_xlabel('meas. angles')
	axes[2].set_ylabel('corrected. angles')
	axes[2].set_xlim(1.1*min_meas, 1.1*max_meas)
	axes[2].plot(meas_angles, ideal_angles, 'bo-', label='ideal vs meas.')
	axes[2].plot(angles, angles + interp_diff(angles), 'r--', label='corrected')
	axes[2].legend()



	plt.show()


	return(interp_diff)




compass_correction = {}
compass_correction['ideal_angles'] = np.array([-3.142, -2.356, -1.571, -0.785, 0.000, 0.785, 1.571, 2.356, 3.142])
compass_correction['meas_angles'] = np.array([1.128, 2.102, -3.043, -2.380, -1.715, -1.060, -0.509, 0.101, 1.109])
compass_correction['min_raw'] = -3.056318240361758
compass_correction['max_raw'] = 3.0952315499358853

getCorrectionInterpolation(compass_correction)

exit()





#
