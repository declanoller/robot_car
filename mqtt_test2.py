import paho.mqtt.client as mqtt #import the client1
from time import sleep
#broker_address="192.168.1.184"
broker_address = 'localhost'
#broker_address="iot.eclipse.org"
print("creating new instance")
client = mqtt.Client("P1") #create new instance
print("connecting to broker")
client.connect(broker_address) #connect to broker
print("Subscribing to topic","house/bulbs/bulb1")
client.subscribe("IR_read")

def on_message(client, userdata, message):
    print("\n\nmessage received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    print("message qos=",message.qos)
    print("message retain flag=",message.retain)


client.on_message=on_message

try:
    client.loop_start()
    while True:
        pass

except:
    client.loop_stop()
