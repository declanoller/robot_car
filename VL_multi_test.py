
import time
import VL53L0X
import RPi.GPIO as GPIO

# GPIO for Sensor 1 shutdown pin ### GRAY WIRE -> BLUE -> PURPLE
sensor1_shutdown = 8
# GPIO for Sensor 2 shutdown pin ### RED -> RED -> BLUE
sensor2_shutdown = 22
# GPIO for Sensor 3 shutdown pin ### BROWN -> GREEN -> BROWN
sensor3_shutdown = 18

GPIO.setwarnings(False)

# Setup GPIO for shutdown pins on each VL53L0X
GPIO.setmode(GPIO.BOARD)
GPIO.setup(sensor1_shutdown, GPIO.OUT)
GPIO.setup(sensor2_shutdown, GPIO.OUT)
GPIO.setup(sensor3_shutdown, GPIO.OUT)

print('Shutting down all')
# Set all shutdown pins low to turn off each VL53L0X
GPIO.output(sensor1_shutdown, GPIO.LOW)
GPIO.output(sensor2_shutdown, GPIO.LOW)
GPIO.output(sensor3_shutdown, GPIO.LOW)

# Keep all low for 500 ms or so to make sure they reset
time.sleep(0.7)

# Create one object per VL53L0X passing the address to give to
# each.
print('Creating tof1 at address 0x29')
tof1 = VL53L0X.VL53L0X(i2c_address=0x29)
print('Turning tof1 on')
GPIO.output(sensor1_shutdown, GPIO.HIGH)
time.sleep(0.1)
print('Changing tof1 to address 0x2a')
tof1.change_address(0x2a)
time.sleep(0.1)

print('Creating tof2 at address 0x29')
tof2 = VL53L0X.VL53L0X(i2c_address=0x29)
print('Turning tof2 on')
GPIO.output(sensor2_shutdown, GPIO.HIGH)
time.sleep(0.1)
print('Changing tof2 to address 0x2b')
tof2.change_address(0x2b)
time.sleep(0.1)

print('Creating tof3 at address 0x29')
tof3 = VL53L0X.VL53L0X(i2c_address=0x29)
print('Turning tof3 on')
GPIO.output(sensor3_shutdown, GPIO.HIGH)
time.sleep(0.1)
print('Changing tof3 to address 0x2c')
tof3.change_address(0x2c)
time.sleep(0.1)


tof1.open()
tof2.open()
tof3.open()

# Set shutdown pin high for the first VL53L0X then
# call to start ranging
#GPIO.output(sensor1_shutdown, GPIO.HIGH)
time.sleep(0.1)
tof1.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)

# Set shutdown pin high for the second VL53L0X then
# call to start ranging
#GPIO.output(sensor2_shutdown, GPIO.HIGH)
time.sleep(0.1)
tof2.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)


# Set shutdown pin high for the second VL53L0X then
# call to start ranging
#GPIO.output(sensor3_shutdown, GPIO.HIGH)
time.sleep(0.1)
tof3.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)





timing = tof1.get_timing()
if timing < 20000:
    timing = 20000
print("Timing %d ms" % (timing/1000))

for count in range(1, 200):

    print()

    distance = tof1.get_distance()
    if distance > 0:
        print("sensor %d - %d mm, %d cm, iteration %d" % (1, distance, (distance/10), count))
    else:
        print("%d - Error" % 1)

    distance = tof2.get_distance()
    if distance > 0:
        print("sensor %d - %d mm, %d cm, iteration %d" % (2, distance, (distance/10), count))
    else:
        print("%d - Error" % 2)

    distance = tof3.get_distance()
    if distance > 0:
        print("sensor %d - %d mm, %d cm, iteration %d" % (3, distance, (distance/10), count))
    else:
        print("%d - Error" % 3)

    time.sleep(timing/1000000.00)

tof3.stop_ranging()
GPIO.output(sensor3_shutdown, GPIO.LOW)
tof2.stop_ranging()
GPIO.output(sensor2_shutdown, GPIO.LOW)
tof1.stop_ranging()
GPIO.output(sensor1_shutdown, GPIO.LOW)

tof1.close()
tof2.close()
tof3.close()


#
