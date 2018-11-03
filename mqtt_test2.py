import paho.mqtt.client as mqtt #import the client1
from time import sleep
broker_address = 'localhost'
client = mqtt.Client("P1") #create new instance
client.connect(broker_address) #connect to broker
print("Subscribing to topic",'IR_read')
client.subscribe("IR_read")

def on_message(client, userdata, message):
    print("\n\nmessage received " ,str(message.payload.decode("utf-8")))

client.on_message=on_message

try:
    client.loop_start()
    while True:
        sleep(0.2)

except:
    client.loop_stop()
