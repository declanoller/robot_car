import paho.mqtt.client as mqtt
import time
import json

class CommMQTT:

    def __init__(self, broker_address='localhost'):

        self.broker_address = broker_address
        self.instance_name = 'RPi'

        print('\nCreating new instance ', self.instance_name)
        self.client = mqtt.Client(self.instance_name) #create new instance

        print('Connecting to broker...')
        self.client.connect(self.broker_address) #connect to broker
        print('Connected.')

        self.client.on_message = self.on_message

        self.IR_topic_name = 'IR_read'
        print('Subscribing to topic', self.IR_topic_name)
        self.client.subscribe(self.IR_topic_name)

        self.latest_reading_IR = None
        self.client.loop_start()
        print('Client loop started.')

    def on_message(self, client, userdata, message):
        payload =  message.payload.decode("utf-8").strip('\n')
        m_in = json.loads(payload) #decode json data
        '''print('payload:', payload)
        print('dict:',m_in)
        print('dict type:',type(m_in))'''
        self.latest_reading_IR = m_in

    def getLatestReadingIR(self):
        return(self.latest_reading_IR)




    def __del__(self):
        self.client.disconnect() #disconnect
        self.client.loop_stop() #stop loop











#
