import matplotlib.pyplot as plt
import numpy as np
import FileSystemTools as fst
from math import pi
from scipy.interpolate import interp1d

meas_angles = np.array([pi, 2.47, 1.85, 1.16, 0.35, -0.57, -1.69, -2.46, -pi])
ideal_angles = np.array([pi, pi*3/4, pi*2/4, pi*1/4, pi*0/4, -pi*1/4, -pi*2/4, -pi*3/4, -pi])

plt.plot(ideal_angles, meas_angles, 'bo-', label='meas. vs ideal')
plt.plot(ideal_angles, ideal_angles, 'ro-', label='ideal vs ideal')
plt.legend()
plt.show()


diff = ideal_angles - meas_angles

interp_diff = interp1d(meas_angles, diff, kind='cubic')
angles = np.linspace(-pi, pi, 50)

plt.plot(meas_angles, diff, 'bo-', label='ideal - meas')
plt.plot(angles, interp_diff(angles), 'r--', label='interp')
plt.legend()
plt.show()



#
