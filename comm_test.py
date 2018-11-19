from CommMQTT import CommMQTT
import time

comm = CommMQTT(broker_address='192.168.1.240')

try:
    while True:
        time.sleep(0.2)
        print('latest reading:',comm.getLatestReadingIR())

except KeyboardInterrupt:
    print('kb exit')

comm.getLatestReadingIR()
