from TOF import TOF
from time import sleep

TOF_forward = TOF(GPIO_SHUTDOWN=8, i2c_address=0x2a)
TOF_left = TOF(GPIO_SHUTDOWN=22, i2c_address=0x2b)
TOF_right = TOF(GPIO_SHUTDOWN=18, i2c_address=0x2c)
TOF_forward.tofOpen()
TOF_left.tofOpen()
TOF_right.tofOpen()
TOF_forward.tofStartRanging()
TOF_left.tofStartRanging()
TOF_right.tofStartRanging()

try:
    while True:

        print('front distance: {:.3f}, left distance: {:.3f}, right distance: {:.3f}, '.format(TOF_forward.distance(), TOF_left.distance(), TOF_right.distance()))
        sleep(0.2)

except:

    print('interrupted')


print('\n\nDeleting TOF objects...')
del TOF_forward
del TOF_left
del TOF_right












#
