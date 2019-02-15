import paho.mqtt.client as mqtt
import time
import json

class MQTTComm:

    def __init__(self, instance_name='Rpi', broker_address='localhost'):

        self.broker_address = broker_address
        self.instance_name = instance_name

        print('\nCreating new instance ', self.instance_name)
        self.client = mqtt.Client(self.instance_name) #create new instance

        print('Connecting to broker...')
        self.client.connect(self.broker_address, 1883) #connect to broker
        print('Connected.')

        self.client.on_message = self.on_message

        self.IR_topic_name = 'IR_read'
        print('Subscribing to topic', self.IR_topic_name)
        self.client.subscribe(self.IR_topic_name)

        self.new_msg = False
        self.latest_reading_IR = None
        self.timeout = 3
        #self.client.loop_start()
        #self.client.loop_forever()
        #print('Client loop started.')

        self.debug_topic_name = 'debug_log'
        self.iteration_topic_name = 'iterate'



    def startLoop(self):
        print('Starting client loop...')
        self.client.loop_forever()
        print('Client loop started.')


    def on_message(self, client, userdata, message):
        self.new_msg = True
        payload =  message.payload.decode("utf-8").strip('\n')
        #print('payload:', payload)
        m_in = json.loads(payload) #decode json data
        #print('dict:',m_in)
        #print('dict type:',type(m_in))
        self.latest_reading_IR = m_in

    def getLatestReadingIR(self):
        start = time.time()
        while True:
            if self.new_msg:
                self.new_msg = False
                return(self.latest_reading_IR)

            if time.time()-start>self.timeout:
                print('MQTT getLatestReadingIR() timed out! returning None')
                return(None)


    def publishDebug(self, msg):
        self.client.publish(self.debug_topic_name, payload=msg)



    def publishIteration(self, msg):
        # Pass this a dict
        json_msg = json.dumps(msg)
        self.client.publish(self.iteration_topic_name, payload=json_msg)




    def __del__(self):
        self.client.disconnect() #disconnect
        self.client.loop_stop() #stop loop











#
