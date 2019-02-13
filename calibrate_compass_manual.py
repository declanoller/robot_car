from Compass import Compass
import numpy as np
from math import pi
import threading
import time
import curses
from Motor import Motor
import FileSystemTools as fst
import json

motor = Motor(left_forward_pin=32, left_reverse_pin=33, right_forward_pin=35, right_reverse_pin=37)
print('Motor object created.')

action_func_list = [motor.goStraight, motor.goBackward, motor.tinyCCW, motor.tinyCW]


compass = Compass(flip_x=True, pi_max=True)
print('Compass object created.')

# Using daemon=True will cause this thread to die when the main program dies.
print('creating compass read loop thread...')
compass_read_thread = threading.Thread(target=compass.readCompassLoop, kwargs={'test_time':'forever', }, daemon=True)
print('starting compass read loop thread...')
compass_read_thread.start()
print('started.')



goal_angle_line = 0
last_action_line = 4
cur_direction_line = 6

min_angle_line = 8
max_angle_line = 9

status_line_1 = 11
status_line_2 = 12




def printLastAction(stdscr, action_str):
    stdscr.addstr(last_action_line, 0, action_str)


def printGoalAngle(stdscr, angle):
    str = 'Move to angle {:.3f} now! Press "a" to save and go to next'.format(angle)
    stdscr.addstr(goal_angle_line, 0, str)


def printMinMaxAngle(stdscr, min_angle, max_angle):
    str = 'Min angle so far: {:.3f}'.format(min_angle)
    stdscr.addstr(min_angle_line, 0, str)
    str = 'Max angle so far: {:.3f}'.format(max_angle)
    stdscr.addstr(max_angle_line, 0, str)


def printCompassDirection(stdscr, angle):
    comp_str = 'Current compass direction: {:.3f}'.format(angle)
    stdscr.addstr(cur_direction_line, 0, comp_str)


def moveCursorRefresh(stdscr):
    stdscr.move(curses.LINES - 1, curses.COLS - 1)
    stdscr.refresh() #Do this after addstr

def DCloop(stdscr):
    #https://docs.python.org/3/howto/curses.html
    #https://docs.python.org/3/library/curses.html#curses.window.clrtobot
    delay_time = 0.01

    ideal_angles = [-pi, -pi*3/4, -pi*2/4, -pi*1/4, 0, pi*1/4, pi*2/4, pi*3/4, pi*4/4]
    meas_angles = []

    save_file = True

    printCompassDirection(stdscr, compass.getCompassDirection())


    ### For figuring out the min/max.
    stdscr.addstr(status_line_1, 0, 'Figuring out min/max. Rotate through full circle.')
    stdscr.addstr(status_line_2, 0, 'Press a to save.')

    raw_compass_readings = []

    while True:
        c = stdscr.getch()

        if c == curses.KEY_LEFT:
            raw_compass_readings.append(compass.getCompassDirection())
            action_func_list[2]()
            time.sleep(delay_time)
            raw_compass_readings.append(compass.getCompassDirection())
            printLastAction(stdscr, 'Pressed Left key, turning CCW')
            printCompassDirection(stdscr, compass.getCompassDirection())
            printMinMaxAngle(stdscr, min(raw_compass_readings), max(raw_compass_readings))
            moveCursorRefresh(stdscr)


        if c == curses.KEY_RIGHT:
            raw_compass_readings.append(compass.getCompassDirection())
            action_func_list[3]()
            time.sleep(delay_time)
            raw_compass_readings.append(compass.getCompassDirection())
            printLastAction(stdscr, 'Pressed Left key, turning CW')
            printCompassDirection(stdscr, compass.getCompassDirection())
            printMinMaxAngle(stdscr, min(raw_compass_readings), max(raw_compass_readings))
            moveCursorRefresh(stdscr)


        if c == curses.KEY_UP:
            raw_compass_readings.append(compass.getCompassDirection())
            action_func_list[0]()
            time.sleep(delay_time)
            raw_compass_readings.append(compass.getCompassDirection())
            printLastAction(stdscr, 'Pressed Up key, going straight')
            printCompassDirection(stdscr, compass.getCompassDirection())
            printMinMaxAngle(stdscr, min(raw_compass_readings), max(raw_compass_readings))
            moveCursorRefresh(stdscr)


        if c == curses.KEY_DOWN:
            raw_compass_readings.append(compass.getCompassDirection())
            action_func_list[1]()
            time.sleep(delay_time)
            raw_compass_readings.append(compass.getCompassDirection())
            printLastAction(stdscr, 'Pressed Down key, going backwards')
            printCompassDirection(stdscr, compass.getCompassDirection())
            printMinMaxAngle(stdscr, min(raw_compass_readings), max(raw_compass_readings))
            moveCursorRefresh(stdscr)


        if c == ord('a'):

            raw_compass_readings.append(compass.getCompassDirection())
            moveCursorRefresh(stdscr)
            break


        elif c == ord('q'):
            break  # Exit the while loop




    # For actually figuring out values.
    stdscr.erase()

    min_raw = min(raw_compass_readings)
    max_raw = max(raw_compass_readings)

    stdscr.addstr(status_line_1, 0, 'Min raw compass reading = {:.2f}'.format(min_raw))
    stdscr.addstr(status_line_2, 0, 'Max raw compass reading = {:.2f}'.format(max_raw))


    cur_angle_index = 0
    printGoalAngle(stdscr, ideal_angles[cur_angle_index])
    printCompassDirection(stdscr, compass.getCompassDirection())

    while True:
        c = stdscr.getch()

        if c == curses.KEY_LEFT:
            action_func_list[2]()
            time.sleep(delay_time)

            printLastAction(stdscr, 'Pressed Left key, turning CCW')
            printCompassDirection(stdscr, compass.getCompassDirection())
            moveCursorRefresh(stdscr)


        if c == curses.KEY_RIGHT:
            action_func_list[3]()
            time.sleep(delay_time)

            printLastAction(stdscr, 'Pressed Left key, turning CW')
            printCompassDirection(stdscr, compass.getCompassDirection())
            moveCursorRefresh(stdscr)


        if c == curses.KEY_UP:
            action_func_list[0]()
            time.sleep(delay_time)

            printLastAction(stdscr, 'Pressed Up key, going straight')
            printCompassDirection(stdscr, compass.getCompassDirection())
            moveCursorRefresh(stdscr)


        if c == curses.KEY_DOWN:
            action_func_list[1]()
            time.sleep(delay_time)

            printLastAction(stdscr, 'Pressed Down key, going backwards')
            printCompassDirection(stdscr, compass.getCompassDirection())
            moveCursorRefresh(stdscr)


        if c == ord('a'):

            meas_angles.append(compass.getCompassDirection())

            cur_angle_index += 1
            if cur_angle_index==len(ideal_angles):
                save_file = True
                break

            stdscr.erase()
            printGoalAngle(stdscr, ideal_angles[cur_angle_index])
            printCompassDirection(stdscr, compass.getCompassDirection())

            #stdscr.addstr(1, 0, 'Pressed r, refreshing')
            moveCursorRefresh(stdscr)


        elif c == ord('q'):
            break  # Exit the while loop



    '''
    should be of the form:

    compass_correction = {}
    compass_correction['ideal_angles'] = np.array([pi, pi*3/4, pi*2/4, pi*1/4, pi*0/4, -pi*1/4, -pi*2/4, -pi*3/4, -pi])
    compass_correction['meas_angles'] = np.array([pi, 2.47, 1.85, 1.16, 0.35, -0.57, -1.69, -2.46, -pi])

    '''
    if save_file:

        print('ideal_angles: ', ideal_angles)
        print('meas_angles: ', meas_angles)


        compass_correction = {}
        compass_correction['ideal_angles'] = ideal_angles
        compass_correction['meas_angles'] = meas_angles
        compass_correction['min_raw'] = min_raw
        compass_correction['max_raw'] = max_raw

        fname = fst.getDateString() + '_compass_cal.json'
        print('writing compass correction to: ', fname)

        with open(fname, 'w+') as f:
            json.dump(compass_correction, f, indent=4)




def directControl():
    print('entering curses loop')
    curses.wrapper(DCloop)
    print('exited curses loop.')





directControl()





'''

SCRAP



        f = open(fst.getDateString() + '_compass_cal.dat', 'w+')
        f.write('compass_correction = {}\n')
        f.write("compass_correction['ideal_angles'] = np.array([{}])\n".format(', '.join(['{:.2f}'.format(x) for x in ideal_angles])))
        f.write("compass_correction['meas_angles'] = np.array([{}])\n".format(', '.join(['{:.2f}'.format(x) for x in meas_angles])))
        f.write("compass_correction['min_raw'] = {:.2f}\n".format(min_raw))
        f.write("compass_correction['max_raw'] = {:.2f}\n".format(max_raw))
        f.close()



'''
























#
