from CommMQTT import CommMQTT
import time

comm = CommMQTT()

try:
    while True:
        time.sleep(0.2)
        print('latest reading:',comm.getLatestReadingIR())

except KeyboardInterrupt:
    print('kb exit')

comm.getLatestReadingIR()
