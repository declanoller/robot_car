from MQTTComm import MQTTComm
import time
import threading

comm = MQTTComm(instance_name='Rpi', broker_address='192.168.1.240')


for i in range(20):

    time.sleep(0.3)
    comm.publishDebug('msg {}'.format(i))
