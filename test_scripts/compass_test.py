from Compass import Compass

compass_correction = {}
compass_correction['ideal_angles'] = np.array([pi, pi*3/4, pi*2/4, pi*1/4, pi*0/4, -pi*1/4, -pi*2/4, -pi*3/4, -pi])
compass_correction['meas_angles'] = np.array([pi, 2.47, 1.85, 1.16, 0.35, -0.57, -1.69, -2.46, -pi])
compass = Compass(compass_correction=compass_correction)
print('Compass object created.')

reading = compass.readCompassLoop(test_time=150)
