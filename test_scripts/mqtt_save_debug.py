import paho.mqtt.client as mqtt
import time
import threading
import FileSystemTools as fst

date_time_string = fst.getDateString()
debug_log_file = 'debug_{}.log'.format(date_time_string)
f = open(debug_log_file, 'w+')
f.write('START\n\n\n')
f.close()

def on_message(client, userdata, message):
    payload =  message.payload.decode("utf-8")
    f = open(debug_log_file, 'a')
    f.write(payload)
    f.close()
    print(payload)



broker_address = '192.168.1.240'
instance_name = 'other_PC'

client = mqtt.Client(instance_name) #create new instance
print('Connecting to broker...')
client.connect(broker_address, 1883) #connect to broker
print('Connected.')
client.on_message = on_message

debug_topic_name = 'debug_log'
print('Subscribing to topic', debug_topic_name)
client.subscribe(debug_topic_name)

loop_thread = threading.Thread(target=client.loop_forever, daemon=True)
loop_thread.start()

try:
    while True:
        pass

except KeyboardInterrupt:
    print('kb exit')
