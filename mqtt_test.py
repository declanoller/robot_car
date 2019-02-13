from MQTTComm import MQTTComm
import time
import threading

comm = MQTTComm(instance_name='Rpi', broker_address='192.168.1.240')
read_num = 0

loop_thread = threading.Thread(target=comm.startLoop, daemon=True)
loop_thread.start()

try:
    while True:
        #time.sleep(0.2)
        print('Getting reading...')
        print('reading number {}: {}'.format(read_num, comm.getLatestReadingIR()))
        read_num += 1

except KeyboardInterrupt:
    print('kb exit')

comm.getLatestReadingIR()
