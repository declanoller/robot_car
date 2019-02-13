from TOF import TOF
from time import sleep

tof1 = TOF(GPIO_SHUTDOWN=8, i2c_address=0x2a)


try:
    while True:

        print('distance: ', tof1.distance())
        sleep(1)

except:

    print('interrupted')















#
