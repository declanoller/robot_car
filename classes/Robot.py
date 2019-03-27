import time
import RPi.GPIO as GPIO
from Motor import Motor
from Sonar import Sonar
from TOF import TOF
from Compass import Compass
from MQTTComm import MQTTComm
from DebugFile import DebugFile
import FileSystemTools as fst

import matplotlib.pyplot as plt
import random
import numpy as np
from math import sin, cos, tan, pi

import threading
import curses

import traceback as tb

'''

-getPosition() sets self.position, which various things need. It is currently
called by:
    -getStateVec() (if calcing position)
    -iterate()
    -reward() (if doing software reward)
    -resetStateValues()

it's somewhat slow (can def notice a diff if you have to call it 10x more), so it
should only be called when needed, but the most recent value should also be used for
important things.

self.position is used by:
    -getStateVec() (if calcing pos)
    -reward() (if doing software reward)
    -iterate() (just for display)
    -addToHist()


'''

class Robot:

    def __init__(self, **kwargs):

        '''
        PINS CURRENTLY BEING USED:
        (these are in terms of GPIO.BOARD, i.e., 1-40.)

        # I2C pins: 3, 5 (SDA, SCL)
        # TOF pins: 8, 22, 18 (front, left, right)
        # Motor pins: 32, 33, 35, 37

        '''

        self.verbose_level = kwargs.get('verbose_level', 1)

        self.init_time = fst.getCurTimeObj()
        self.r = 0
        self.initial_iteration = kwargs.get('initial_iteration', 0)
        self.iteration = self.initial_iteration

        #GPIO Mode (BOARD / BCM)
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)

        self.motor_enable = kwargs.get('motor_enable', True)
        self.TOF_enable = kwargs.get('TOF_enable', True)
        self.compass_enable = kwargs.get('compass_enable', True)
        self.MQTT_enable = kwargs.get('MQTT_enable', True)
        self.arena_mode = kwargs.get('arena_mode', True)
        self.debug_enable = kwargs.get('debug_enable', True)

        self.df = DebugFile(notes='Trying with TOF sensors', enabled=self.debug_enable)
        self.date_time_string = self.df.getDateString()

        self.setupMQTT(**kwargs)

        self.print('************************* In function: {}()'.format('init'), 2)
        self.print('motor enable: {}'.format(self.motor_enable), 2)
        self.print('TOF enable: {}'.format(self.TOF_enable), 2)
        self.print('compass enable: {}'.format(self.compass_enable), 2)
        self.print('MQTT enable: {}'.format(self.MQTT_enable), 2)

        self.save_hist = kwargs.get('save_hist', False)

        self.distance_meas_enable = (self.TOF_enable)
        all_arena_meas_enabled = self.motor_enable and self.distance_meas_enable and self.MQTT_enable and self.compass_enable
        if self.arena_mode:
            assert all_arena_meas_enabled, 'Arena mode enabled, but either motor/dist/MQTT/compass not enabled.'

        self.setupMotor(**kwargs)

        self.setupTOF(**kwargs)

        self.setupCompass(**kwargs)

        self.setupArena(**kwargs)





    ########### Functions that the Agent class expects.


    def getStateVec(self):

        if self.arena_mode:
            target_pos = self.target_positions[self.current_target]

            if self.state_type == 'position':
                self.position, compass_reading = self.getPosition()
                normed_angle = compass_reading/pi
                return(np.concatenate((self.position, [normed_angle], target_pos)))
            if self.state_type == 'distances':
                d1, d2, d3 = self.getDistanceMeas()
                compass_reading = self.getCompassDirection()
                normed_angle = compass_reading/pi
                return(np.concatenate(([d1, d2, d3], [normed_angle], target_pos)))

        else:
            return(np.zeros(5))



    def iterate(self, action):

        self.print('********************* In function: iterate()', 3)
        self.getPosition() # just so it has the latest info
        if self.arena_mode:
            iter_str = 'iterate() {}, elapsed time=({}). R_avg={:.4f} --> act={}. pos=({:.2f}, {:.2f}), angle={:.2f}, target={}, targets={}'.format(
            self.iteration,
            fst.getTimeDiffStr(self.init_time),
            self.R_tot/max(1, self.iteration),
            action,
            *self.position,
            self.angle,
            self.current_target,
            [int(x) for x in self.getTriggerList()])

            self.print(iter_str, 1)
            iter_dict = {}
            iter_dict['iteration'] = self.iteration
            iter_dict['position'] = self.position.tolist()
            iter_dict['angle'] = float(self.angle)
            iter_dict['current_target'] = int(self.current_target)
            iter_dict['trigger_list'] = [int(x) for x in self.getTriggerList()]
            iter_dict['d1'] = self.d1
            iter_dict['d2'] = self.d2
            iter_dict['d3'] = self.d3

            self.comm.publishIteration(iter_dict)

        self.doAction(action)
        self.iteration += 1
        self.print('did action {}'.format(action), 2)

        if self.arena_mode:
            self.r = self.reward()
            self.R_tot += self.r
            self.addToHist()
            return(self.r, self.getStateVec())



    def targetTriggered(self, target_num):
        # Right now this is only for figuring out if it's close to a target
        # by calculation, not sensing (with the IR sensors).
        #
        # 0 indexed.
        # returns true if dist. is below reward_distance_thresh, false otherwise.
        # Assumes self.position hasn't changed (or it will be pretty expensive to
        # run several times each iteration.)
        return(np.linalg.norm(self.position - self.target_positions[target_num]) < self.reward_distance_thresh)


    def reward(self):
        if self.reward_method == 'MQTT':
            current_target_triggered = (int(self.pollTargetServer()[self.current_target]) == 1)
        else:
            self.position, angle = self.getPosition()
            current_target_triggered = self.targetTriggered(self.current_target)

        if current_target_triggered:
            self.print('Current target {} reached! Calling resetTarget()'.format(self.current_target), 1)
            self.resetTarget()
            return(1.0)
        else:
            return(-0.01)


    def initEpisode(self):
        self.print('in function initEpisode()', 2)
        self.resetTarget()
        self.resetStateValues()


    def resetStateValues(self):

        self.print('in function resetStateValues()', 2)
        self.position, self.angle = self.getPosition()
        self.last_action = 0
        self.iteration = self.initial_iteration
        self.start_time = time.time()


        self.pos_hist = np.array([self.position])
        self.angle_hist = np.array([self.angle])
        self.action_hist = np.array([self.last_action])
        self.target_hist = np.array([self.current_target])
        self.t = np.array([0])
        self.iterations = np.array([self.iteration])
        self.r_hist = np.array([0])
        self.R_tot = 0




    ########################### Functions for interacting with the environment.


    def resetTarget(self):
        # This makes it so it won't give it one of the adjacent targets either.
        other_targets = [i for i in self.valid_targets if (i!=self.current_target and (i != (self.current_target-1)%self.N_targets) and (i != (self.current_target+1)%self.N_targets))]
        #other_targets = [i for i in self.valid_targets if (i!=self.current_target)]
        self.current_target = random.choice(other_targets)
        self.current_target_pos = self.target_positions[self.current_target]
        self.print('Target reset in resetTarget(). Current target is now {}.'.format(self.current_target), 1)


    def getTriggerList(self):
        # This gives you the binary list of the targets that have been triggered,
        # whether you're using the MQTT reward method or just calcing it yourself.
        if self.reward_method == 'MQTT':
            return(self.pollTargetServer())
        else:
            # I know it's fucking stupid to do this, but right now pollTargetServer
            # returns a string and I just wanna drop this function in for it.
            return([str(int(self.targetTriggered(i))) for i in self.valid_targets])


    def pollTargetServer(self):
        mqtt_reading = self.comm.getLatestReadingIR()
        if mqtt_reading is None:
            self.print('No mqtt reading, returning all 0.', 2)
            return(['0']*self.N_targets)

        return(mqtt_reading['IR_reading'].split())


    def doAction(self, action):
        assert self.motor_enable, 'Motor not enabled! crashing.'
        # Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.last_action = action
        self.action_func_list[action]()


    def angleToPiBds(self, ang):

        # Just converts any angle to its equivalent in [-pi, pi] bounds.

        ang = ang%(2*pi) # This will make it into [0, 2*pi] bds
        if ang > pi:
            return(ang - 2*pi)
        else:
            return(ang)


    def touchingSameWall(self, a, a_theta, b, b_theta):
        #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
        #If any are, then it returns the coord and the index of it. Otherwise, returns None.
        #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
        #same_coord_acc_x = abs((x1 - x2)/self.wall_length)
        same_coord_error_x = abs(x1 - x2)/(0.5*(abs(x1) + abs(x2)))

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        #same_coord_error_y = abs((y1 - y2)/self.wall_length)
        same_coord_error_y = abs(y1 - y2)/(0.5*(abs(y1) + abs(y2)))

        if same_coord_error_x < same_coord_error_y:
            return(x1, 0, same_coord_error_x)
        else:
            return(y1, 1, same_coord_error_y)


    def touchingOppWall(self, a, a_theta, b, b_theta):
        #Returns index of two that are touching opp walls, None otherwise.
        #Also returns the coordinate we're sure about now, which is the negative of the
        #negative of the pair that makes up the span.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        span_x = abs(x1 - x2)
        #span_x = abs(x1) + abs(x2)
        #print('span x={}'.format(span))
        #if abs((span - self.wall_length)/(0.5*(span + self.wall_length))) < self.dist_meas_percent_tolerance:
        span_error_x = abs((span_x - self.wall_length)/self.wall_length)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        span_y = abs(y1 - y2)
        #span_y = abs(y1) + abs(y2)
        span_error_y = abs((span_y - self.wall_length)/self.wall_length) # Lower is better

        if span_error_x < span_error_y:
            if x1 < 0:
                if abs(x1) < abs(x2):
                    return(-x1, 0, span_error_x)
                else:
                    return(self.wall_length - x2, 0, span_error_x)
            else:
                if abs(x1) < abs(x2):
                    return(self.wall_length - x1, 0, span_error_x)
                else:
                    return(-x2, 0, span_error_x)
        else:
            if y1 < 0:
                if abs(y1) < abs(y2):
                    return(-y1, 1, span_error_y)
                else:
                    return(self.wall_length - y2, 1, span_error_y)
            else:
                if abs(y1) < abs(y2):
                    return(self.wall_length - y1, 1, span_error_y)
                else:
                    return(-y2, 1, span_error_y)


    def posAngleToCollideVec(self, pos, ang):
        #
        # This takes a proposed position and angle, and returns the vector
        # that would collide with the wall it would hit at that angle.
        #
        # This assumes *lower left* origin coords.
        #
        # The angle bds here are the 4 angle bounds that determine
        # for a given x,y which wall the ray in that direction would collide with.
        # The order is: top wall, left wall, bottom wall, right wall

        ang = self.angleToPiBds(ang)

        x, y = pos

        angle_bds = [
        np.arctan2(self.wall_length-y, self.wall_length-x),
        np.arctan2(self.wall_length-y, -x),
        np.arctan2(-y, -x),
        np.arctan2(-y, self.wall_length-x),
        ]

        ray_x, ray_y = 0, 0

        if (ang >= angle_bds[0]) and (ang < angle_bds[1]):
            ray_y = self.wall_length - y
            if abs(ang) > 0.1 :
                ray_x = ray_y/tan(ang) #, self.wall_length*np.sign(tan(ang)))
            else:
                if np.sign(tan(ang)) > 0:
                    ray_x = self.wall_length - x
                else:
                    ray_x = -x

        elif (ang >= angle_bds[1]) or (ang < angle_bds[2]):
            ray_x = -x
            ray_y = ray_x*tan(ang)

        elif (ang >= angle_bds[2]) and (ang < angle_bds[3]):
            ray_y = -y
            if abs(ang) > 0.1 :
                ray_x = ray_y/tan(ang) #, self.wall_length*np.sign(tan(ang)))
            else:
                if np.sign(tan(ang)) > 0:
                    ray_x = self.wall_length - x
                else:
                    ray_x = -x

        elif (ang >= angle_bds[3]) and (ang < angle_bds[0]):
            ray_x = self.wall_length - x
            ray_y = ray_x*tan(ang)

        return(np.array([ray_x, ray_y]))


    def formatPosTuple(self, tup):

        if tup is None:
            return(None)
        else:
            return('({:.3f}, {}, {:.3f})'.format(*tup))


    def cornerOriginToCenterOrigin(self, pos):

        # This changes it so if your coords are so that the origin is in the
        # bottom left hand corner, now it's in the middle of the arena.

        center_origin_pos = pos + self.bottom_corner
        return(center_origin_pos)


    def calculatePosition(self, d1, d2, d3, theta):

        self.print('************************* In function: {}()'.format('calculatePosition'), 3)
        #This uses some...possibly sketchy geometry, but I think it should work
        #generally, no matter which direction it's pointed in.
        #
        #There are 3 possibilities for the configuration: two sonars are hitting the same wall,
        #two sonars are hitting opposite walls, or both.
        #If it's one of the first two, the position is uniquely specified, and you just have to
        #do the painful geometry for it. If it's the third, it's actually not specified, and you
        #can only make an educated guess within some range.
        #
        #d1 is the front sonar, d2 the left, d3 the right. From now on they will be in units of METERS.
        #
        # For clarity's sake: this is set up so you have x in the "horizontal"
        # direction (the direction of theta = 0), and y in the vertical direction.
        # theta increase CCW, like typical in 2D polar coords.
        # Here, it will return the coords where the origin is the center of the arena.
        #
        #


        d1_vec = d1*np.array([cos(theta + 0), sin(theta + 0)])
        d2_vec = d2*np.array([cos(theta + pi/2), sin(theta + pi/2)])
        d3_vec = d3*np.array([cos(theta - pi/2), sin(theta - pi/2)])

        pair12 = [d1, theta, d2, theta + pi/2]
        pair23 = [d2, theta + pi/2, d3, theta - pi/2]
        pair13 = [d1, theta, d3, theta - pi/2]

        pair12_same = self.touchingSameWall(*pair12)
        pair13_same = self.touchingSameWall(*pair13)

        pair12_opp = self.touchingOppWall(*pair12)
        pair23_opp = self.touchingOppWall(*pair23)
        pair13_opp = self.touchingOppWall(*pair13)

        self.print('Same walls: pair12_same={}, pair13_same={}'.format(
        self.formatPosTuple(pair12_same),
        self.formatPosTuple(pair13_same)), 2)
        self.print('Opp walls: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(
        self.formatPosTuple(pair12_opp),
        self.formatPosTuple(pair23_opp),
        self.formatPosTuple(pair13_opp)), 2)

        self.print('\n', 2)
        self.print('d1: ({:.3f}, {:.3f}), d2: ({:.3f}, {:.3f}), d3: ({:.3f}, {:.3f})\n'.format(*d1_vec, *d2_vec, *d3_vec), 2)

        errors = [
        ('12', pair12_same, 'same'),
        ('13', pair13_same, 'same'),
        ('12', pair12_opp, 'opp'),
        ('23', pair23_opp, 'opp'),
        ('13', pair13_opp, 'opp')
        ]


        pos_err_list = []

        for err in errors:

            sol = -self.bottom_corner # Puts it at (.65, .65), lower left origin (so center of arena)
            pair_label = err[0]
            dist, coord, acc = err[1]
            match_label = err[2]

            self.print('----------  {}, {}\t{}'.format(pair_label, match_label, self.formatPosTuple(err[1])), 2)
            if match_label == 'same':
                #Sets the coordinate we've figured out.
                if dist>=0:
                    sol[coord] = self.wall_length - dist
                else:
                    sol[coord] = -dist

            if match_label == 'opp':
                #The dist should already be positive.
                sol[coord] = dist


            if pair_label == '12':
                other_ray = d3_vec

            if pair_label == '23':
                other_ray = d1_vec

            if pair_label == '13':
                other_ray = d2_vec

            #This is the other coord we don't have yet, which works for either case.
            other_coord = abs(1 - coord)
            other_dist = other_ray[other_coord]

            self.print('dist={:.3f}, coord={}, other_coord={}, other_ray=({:.3f}, {:.3f})'.format(
            dist, coord, other_coord, *other_ray), 2)

            if other_dist>=0:
                sol[other_coord] = self.wall_length - other_dist
            else:
                sol[other_coord] = -other_dist

            d1_collide_vec = self.posAngleToCollideVec(sol, theta + 0)
            d2_collide_vec = self.posAngleToCollideVec(sol, theta + pi/2)
            d3_collide_vec = self.posAngleToCollideVec(sol, theta - pi/2)

            self.print('d1_col: ({:.3f}, {:.3f}), d2_col: ({:.3f}, {:.3f}), d3_col: ({:.3f}, {:.3f})'.format(
            *d1_collide_vec, *d2_collide_vec, *d3_collide_vec), 2)

            # Right now just adding theirs norms... might be a better way of doing
            # this but this seems to work for now.
            tot_err = np.linalg.norm(d1_vec - d1_collide_vec) + np.linalg.norm(d2_vec - d2_collide_vec) + np.linalg.norm(d3_vec - d3_collide_vec)

            pos = self.cornerOriginToCenterOrigin(sol)
            pos_err_list.append([pos, tot_err, pair_label, match_label])

            self.print('{}, {} finds: pos ({:.3f}, {:.3f}), err ({:.3f})'.format(
                                                                            pair_label,
                                                                            match_label,
                                                                            *pos,
                                                                            tot_err), 2)

        self.print('\n', 2)

        lowest_error_tuple = min(pos_err_list, key=lambda x: x[1])
        self.print('Pos ({:.3f}, {:.3f}) has lowest err ({:.3f}) with {}, {}'.format(
                                                                        *(lowest_error_tuple[0]),
                                                                        lowest_error_tuple[1],
                                                                        lowest_error_tuple[2],
                                                                        lowest_error_tuple[3]), 2)

        pos = lowest_error_tuple[0]
        self.print('pos. calcd in calcPosition=({:.3f}, {:.3f})'.format(*pos), 2)
        return(pos)



    def coordInBds(self, coord):
        # This calculates if an (x,y) np array is in the bounds (plus a little margin)
        # of the arena, with center origin coords. (So, it will consider (0.7, 0.1) to be
        # in bounds even though it's technically out.)
        if abs(coord[0]) > 1.1*self.wall_length/2:
            return(False)
        if abs(coord[1]) > 1.1*self.wall_length/2:
            return(False)

        return(True)


    def getPosition(self):

        self.print('************************* In function: {}()'.format('getPosition'), 2)

        assert self.distance_meas_enable, 'No distance sensors enabled! exiting.'
        assert self.compass_enable, 'No compass! exiting.'

        for attempt in range(self.N_pos_attempts):

            self.print('Attempt #{}...'.format(attempt), 2)
            d1, d2, d3 = self.getDistanceMeas()
            self.d1, self.d2, self.d3 = d1, d2, d3
            compass_reading = self.getCompassDirection() #From now on, the function will prepare and scale everything.

            self.print('Raw measurements: d1={:.3f}, d2={:.3f}, d3={:.3f}, angle={:.3f}'.format(
            d1, d2, d3, compass_reading), 2)

            self.position = self.calculatePosition(d1, d2, d3, compass_reading)
            self.angle = compass_reading

            if self.coordInBds(self.position):
                return(self.position, compass_reading)
            else:
                self.print('Calculated position ({:.3f}, {:.3f}) is out of bounds!!'.format(*self.position), 2)

        get_pos_str = '\n\n\nTried {} attempts in getPos(), all failed. Using (0,0).\n\n'.format(self.N_pos_attempts)
        self.print(get_pos_str, 1)

        self.position = np.array([0,0])
        return(self.position, compass_reading)


    def getCompassDirection(self):

        assert self.compass_enable, 'Compass not enabled in getCompassDirection()!'

        return(self.compass.getCompassDirection())


    def getDistanceMeas(self):

        # This is for interfacing to whatever distance measuring thing you're using, sonar or TOF.
        # It only gets the dist. measures, so you can run it not in arena_mode.

        assert self.distance_meas_enable, 'No distance sensors enabled! exiting.'

        if self.TOF_enable:
            return(self.getTOFMeas())



    def getTOFMeas(self):

        time.sleep(0.1)
        front_sensor_offset = 0.13
        left_sensor_offset = 0.03
        right_sensor_offset = 0.03
        d1 = self.TOF_forward.distance() + front_sensor_offset
        d2 = self.TOF_left.distance() + left_sensor_offset
        d3 = self.TOF_right.distance() + right_sensor_offset

        if self.arena_mode:
            max_dist = self.wall_length*1.3
        else:
            max_dist = 1000000
        return(min(d1, max_dist), min(d2, max_dist), min(d3, max_dist))


    def positionOutOfBounds(self, pos):

        x = pos[0]
        y = pos[1]

        if x<self.xlims[0] or x>self.xlims[1]:
            return(True)
        if y<self.ylims[0] or y>self.ylims[1]:
            return(True)

        return(False)


    def positionPercentOutOfBounds(self, pos_percent):

        # This is just for testing whether it's in bounds, in terms of percent, where
        # 0 is one wall and 1 is the other.
        x = pos_percent[0]
        y = pos_percent[1]

        if x<0 or x>1:
            return(True)
        if y<0 or y>1:
            return(True)

        return(False)


    ################# Functions for interacting directly with the robot.


    def testAllDevices(self, test_duration = 2):

        #test_duration = 2

        if self.motor_enable:
            self.print('testing motor!', 1)
            #self.motor.wheelTest(test_time=2)

        if self.TOF_enable:
            self.print('testing TOF! (front)', 1)
            self.TOF_forward.distanceTestLoop(test_time=test_duration)
            self.print('testing TOF! (left)', 1)
            self.TOF_left.distanceTestLoop(test_time=test_duration)
            self.print('testing TOF! (right)', 1)
            self.TOF_right.distanceTestLoop(test_time=test_duration)

        if self.compass_enable:
            self.print('testing compass!', 1)
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            self.compass.readCompassLoop(test_time=test_duration, print_readings=True)


    def DCloop(self, stdscr):
        #https://docs.python.org/3/howto/curses.html
        #https://docs.python.org/3/library/curses.html#curses.window.clrtobot
        self.print('************************* In function: {}()'.format('DCloop'), 1)
        self.print('Size of curses window: LINES={}, COLS={}'.format(curses.LINES, curses.COLS), 1)
        delay_time = 0.1

        move_str_pos = [0, 6]

        self.drawStandard(stdscr)

        while True:
            c = stdscr.getch()

            if c == curses.KEY_LEFT:
                self.print('Pressed Left key, turning CCW', 3)
                self.iterate(2)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Left key, turning CCW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_RIGHT:
                self.print('Pressed Right key, turning CW', 3)
                self.iterate(3)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Right key, turning CW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_UP:
                self.print('Pressed Up key, going straight', 3)
                self.iterate(0)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Up key, going straight')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_DOWN:
                self.print('Pressed Down key, going backwards', 3)
                self.iterate(1)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Down key, going backwards')
                self.moveCursorRefresh(stdscr)


            if c == ord('r'):
                self.print('Pressed r, refreshing', 3)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed r, refreshing')
                self.moveCursorRefresh(stdscr)


            elif c == ord('q'):
                self.print('you pressed q! exiting', 1)
                break  # Exit the while loop


    def moveCursorRefresh(self, stdscr):
        stdscr.move(curses.LINES - 1, curses.COLS - 1)
        stdscr.refresh() #Do this after addstr


    def directControl(self):
        self.print('entering curses loop', 1)
        curses.wrapper(self.DCloop)
        self.print('exited curses loop.', 1)



    def redrawBox(self, stdscr, box_info, pos, angle):
        # From now on, pass the pos parameter as a percent of the box width in each
        # direction, so you don't have to deal with scaling things outside of this func.
        side_symbol = '-'
        (box_coord_y, box_coord_x, box_side_y, box_side_x) = box_info
        stdscr.addstr(box_coord_y - 1, box_coord_x, 2*side_symbol*box_side_x)
        stdscr.addstr(box_coord_y + box_side_y, box_coord_x, 2*side_symbol*box_side_x)
        stdscr.addstr(box_coord_y + box_side_y + 1, box_coord_x - 12, '(-0.62, -0.62)')
        stdscr.addstr(box_coord_y - 1, box_coord_x + 2*box_side_x + 1, '(0.625, 0.625)')

        for i in range(box_side_y):
            stdscr.addstr(box_coord_y + i, box_coord_x - 1, '|')
            stdscr.addstr(box_coord_y + i, box_coord_x + 2*box_side_x, '|')

        # Note that only the x is scaled here.
        arrow_list = ['⮕','⬈','↑','⬉','⟵','⬋','↓','↘']
        angle_ind = int(divmod((angle + pi/8)%(2*pi), pi/4)[0])
        if self.positionPercentOutOfBounds(pos):
            self.print('Position PERCENT of box {} out of bounds in redrawBox(), setting to 0.5,0.5 for drawing.'.format(pos), 3)
            pos = [0.5, 0.5]
        stdscr.addstr(box_coord_y + -1 + (box_side_y - int(pos[1]*box_side_y)), box_coord_x + 1 + int(pos[0]*2*box_side_x), arrow_list[angle_ind])


    def drawStandard(self, stdscr):
        stdscr.erase()
        self.print('************************* In function: {}()'.format('drawStandard'), 3)

        if self.distance_meas_enable:
            self.print('Getting distance info', 3)
            d1, d2, d3 = self.getDistanceMeas()
            info_str = 'Distance meas: (straight = {:.2f}, left = {:.2f}, right = {:.2f})'.format(d1, d2, d3)
            self.print(info_str, 3)
            stdscr.addstr(0, 0,  info_str)

        if self.compass_enable:
            self.print('Getting compass info', 3)
            compass_reading = self.getCompassDirection() #From now on, the function will prepare and scale everything.
            info_str = 'Compass meas: ({:.2f})'.format(compass_reading)
            self.print(info_str, 3)
            stdscr.addstr(2, 0,  info_str)


        if self.arena_mode:

            self.print('Getting position info', 3)

            #Draw box, with best position guess
            box_side_x = 25
            box_side_y = box_side_x
            box_coord_y = 9
            box_coord_x = 45

            # getPosition should return the position, where (0,0) is the center of
            # the arena. box_pos is the integer pair for the drawn box indices.

            try:
                raw_pos, angle = self.getPosition()
                pos_str = '({:.2f}, {:.2f})'.format(*raw_pos)
                self.print('Got position info: {}'.format(pos_str), 3)
                stdscr.addstr(4, 0, 'Estimated position: {}'.format(pos_str))

                box_percent_pos = (raw_pos - self.bottom_corner)/self.wall_length

            except:
                box_percent_pos = np.array([0.5, 0.5])
                angle = 0

            self.print('drawing agent pos (o) at percent ({:.3f}, {:.3f})'.format(*box_percent_pos), 3)
            box_info = (box_coord_y, box_coord_x, box_side_y, box_side_x)
            self.redrawBox(stdscr, box_info, box_percent_pos, angle)


        if self.MQTT_enable:
            self.print('Getting IR target info...', 3)
            reading_str = ' '.join(self.getTriggerList())
            IR_read = 'IR target reading:  ' + reading_str
            self.print('Got IR target info: {}'.format(reading_str), 3)
            stdscr.addstr(8, 0,  IR_read)

        stdscr.addstr(curses.LINES - 1, 0,  'Press q or Esc to quit')

        stdscr.refresh() #Do this after addstr
        self.print('end of drawStandard()\n\n', 3)


    ############ Bookkeeping functions

    def print(self, print_str, str_verb_level=2):
        # This prints and takes the verbosity level into account,
        # as well as if it should write to the debug file.
        # This only accepts strings, so don't use it exactly as you'd
        # use the normal print() function.
        #
        # There will be a verbose_level for the whole program, that
        # the user sets, that determines how much output they want to
        # see. Then, for each message, there will be a verbose_level
        # that dictates how important the message is.
        #
        # The level the user specifies is such that the higher it is,
        # the more output they'll see. 0 is seeing nothing, 2 is seeing everything.
        #
        # The one you pass to this function determines how important the message is;
        # str_verb_level=0 means that it will print no matter what (overriding even
        # the user specified level 0, so use it rarely, just for errors/etc),
        # and higher up is less important.
        #
        # So usually I'll have self.verbose_level=1, and things like the iteration
        # info will have str_verb_level=1 also, but less important stuff will be 2.
        # The default for this fn will be 2. Pass it 3 if you want it to never print
        # the thing, just pass it to writeToDebug().

        if str_verb_level <= self.verbose_level:
            print(print_str)

        if hasattr(self, 'df'):
            if self.debug_enable:
                self.df.writeToDebug(print_str)
        else:
            print('something wrong in self.print(), either df isnt set up yet or debug_enable=False.')




    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.position]))
        self.angle_hist = np.concatenate((self.angle_hist, [self.angle]))
        self.iterations = np.concatenate((self.iterations, [self.iteration]))
        self.t = np.concatenate((self.t, [time.time() - self.start_time]))
        self.r_hist = np.concatenate((self.r_hist, [self.r]))
        # Only use self.r right after you've set self.r = self.reward()
        self.action_hist = np.concatenate((self.action_hist, [self.last_action]))
        self.target_hist = np.concatenate((self.target_hist, [self.current_target]))


    def saveHist(self, **kwargs):

        all_hist = np.concatenate((
        np.expand_dims(self.iterations, axis=1),
        np.expand_dims(self.t, axis=1),
        self.pos_hist,
        np.expand_dims(self.angle_hist, axis=1),
        np.expand_dims(self.r_hist, axis=1),
        np.expand_dims(self.action_hist, axis=1).astype('float32'),
        np.expand_dims(self.target_hist, axis=1).astype('float32'),
        ), axis=1)

        self.print('allhist shape: {}'.format(all_hist.shape), 1)
        header = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('iter', 't', 'x', 'y', 'ang', 'r', 'action', 'target')

        default_fname = 'all_hist_' + self.date_time_string + '.txt'
        fname = kwargs.get('fname', default_fname)
        self.print('\nSaving robot hist to {}'.format(fname), 1)
        np.savetxt(fname, all_hist, header=header, fmt='%.3f', delimiter='\t')


################################# Setup/delete functions


    def setupCompass(self, **kwargs):

        if self.compass_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.compass_correction_file = kwargs.get('compass_correction_file', None)

            self.compass = Compass(compass_correction_file=self.compass_correction_file, pi_max=True, flip_x=True)
            self.print('Compass object created.', 2)

            # Using daemon=True will cause this thread to die when the main program dies.
            self.print('creating compass read loop thread...', 2)
            self.compass_read_thread = threading.Thread(target=self.compass.readCompassLoop,
            kwargs={'test_time':'forever', },
            daemon=True)
            self.print('starting compass read loop thread...', 2)
            self.compass_read_thread.start()
            self.print('started.', 2)


    def setupMotor(self, **kwargs):

        if self.motor_enable:
            self.motor = Motor(left_forward_pin=32, left_reverse_pin=33, right_forward_pin=35, right_reverse_pin=37)
            self.print('Motor object created.', 2)
            #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
            self.N_actions = 4
            # This determines the order of the action indices (i.e., 0 is straight, 1 is backward, etc)
            self.action_func_list = [self.motor.goStraight, self.motor.goBackward, self.motor.turn90CCW, self.motor.turn90CW]



    def setupMQTT(self, **kwargs):
        # Set up MQTT first, so it can be passed to DebugFile, for debugging purposes.
        self.comm = None
        if self.MQTT_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.comm = MQTTComm(broker_address='192.168.1.240')
            self.print('MQTTComm object created.', 2)
            self.print('Starting MQTT loop...', 2)
            self.MQTT_loop_thread = threading.Thread(target=self.comm.startLoop, daemon=True)
            self.MQTT_loop_thread.start()
            self.print('MQTT loop started.', 2)


    def setupTOF(self, **kwargs):

        if self.TOF_enable:
            self.print('Creating TOF objects...', 2)
            self.TOF_forward = TOF(GPIO_SHUTDOWN=8, i2c_address=0x2a)
            self.TOF_left = TOF(GPIO_SHUTDOWN=22, i2c_address=0x2b)
            self.TOF_right = TOF(GPIO_SHUTDOWN=18, i2c_address=0x2c)
            self.TOF_forward.tofOpen()
            self.TOF_left.tofOpen()
            self.TOF_right.tofOpen()
            self.TOF_forward.tofStartRanging()
            self.TOF_left.tofStartRanging()
            self.TOF_right.tofStartRanging()
            self.print('TOF objects created.', 2)



    def setupArena(self, **kwargs):

        if self.arena_mode:

            # All units here are in meters, and with respect to the origin being
            # in the center of the arena.
            self.N_pos_attempts = 5
            self.wall_length = 1.25
            self.xlims = np.array([-self.wall_length/2, self.wall_length/2])
            self.ylims = np.array([-self.wall_length/2, self.wall_length/2])
            self.position = np.array([0.5*(max(self.xlims) + min(self.xlims)), 0.5*(max(self.ylims) + min(self.ylims))])
            self.last_position = self.position
            self.bottom_corner = np.array([self.xlims[0], self.ylims[0]])
            self.lims = np.array((self.xlims,self.ylims))
            self.robot_draw_rad = self.wall_length/20.0
            self.target_draw_rad = self.robot_draw_rad

            self.dist_meas_percent_tolerance = 0.05

            self.target_positions = np.array([
            [.19, 0],
            [.55, 0],
            [self.wall_length, .21],
            [self.wall_length, .65],
            [.97, self.wall_length],
            [.58, self.wall_length],
            [0, 1.02],
            [0, .60]])
            # This makes it so the target positions are w.r.t. the origin at the center.
            self.target_positions = np.array([self.cornerOriginToCenterOrigin(pos) for pos in self.target_positions])
            #These will be the positions in meters, where (0,0) is the center of the arena.

            self.N_targets = len(self.target_positions)
            self.current_target = 0
            self.print('Current target is: {}'.format(self.current_target), 2)

            # This determines whether we're going to get the rewards from the IR
            # sensors, or just calc. whether we're close enough in distance.
            self.reward_method = kwargs.get('reward_method', 'MQTT')
            if self.reward_method == 'MQTT':
                #This is in terms of x and y, from the bottom left corner.
                # This is because right now, #2 isn't working...
                # These are counting starting from 1
                self.valid_targets = np.array([1, 2, 3, 6, 8])
                # This makes it so they're 0 indexed now
                self.valid_targets -= 1

                # Set actual target. Only work if MQTT enabled but we'll only be
                # here if arena_mode is enabled.
                test_IR_read = self.pollTargetServer()
                self.print('test IR read: {}'.format(test_IR_read), 2)
                assert len(test_IR_read)==self.N_targets, 'Number of targets ({}) returned from MQTTComm doesn\'t match N_targets ({}) '.format(len(test_IR_read), self.N_targets)

            else:
                # In this case, we're just gonna give the reward directly, based on its distance.
                self.reward_distance_thresh = 0.15
                self.valid_targets = np.array(list(range(8)))

            self.state_type = kwargs.get('state_type', 'position')

            self.N_state_terms = len(self.getStateVec())
            self.resetStateValues()
            self.print('Robot has a state vec of length: {}'.format(self.N_state_terms), 2)


    def __del__(self):

        del self.motor
        time.sleep(0.5)
        if self.TOF_enable:
            self.print('\n\nDeleting TOF objects...', 1)
            del self.TOF_forward
            del self.TOF_left
            del self.TOF_right
        time.sleep(0.5)
        GPIO.cleanup()
