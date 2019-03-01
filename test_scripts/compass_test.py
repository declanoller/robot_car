from Compass import Compass
import numpy as np
from math import pi
import threading
import time

compass_correction = {}
compass_correction['ideal_angles'] = np.array([pi, pi*3/4, pi*2/4, pi*1/4, pi*0/4, -pi*1/4, -pi*2/4, -pi*3/4, -pi])
compass_correction['meas_angles'] = np.array([pi, 2.47, 1.85, 1.16, 0.35, -0.57, -1.69, -2.46, -pi])
#compass = Compass(compass_correction=compass_correction)
compass = Compass(raw_reading=True)
print('Compass object created.')

# Using daemon=True will cause this thread to die when the main program dies.
print('creating compass read loop thread...')
compass_read_thread = threading.Thread(target=compass.readCompassLoop, kwargs={'test_time':'forever', }, daemon=True)
print('starting compass read loop thread...')
compass_read_thread.start()
print('started.')



try:
    while True:

        print('Compass reading: {:.3f}'.format(compass.getCompassDirection()))

except:

    print('quitting')
