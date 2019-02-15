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


class Robot:

    def __init__(self, **kwargs):

        '''
        PINS CURRENTLY BEING USED:
        (these are in terms of GPIO.BOARD, i.e., 1-40.)

        # I2C pins: 3, 5 (SDA, SCL)
        # TOF pins: 8, 22, 18 (front, left, right)
        # Motor pins: 32, 33, 35, 37

        '''

        self.init_time = fst.getCurTimeObj()

        self.initial_iteration = kwargs.get('initial_iteration', 0)

        #GPIO Mode (BOARD / BCM)
        GPIO.cleanup()
        #print('GPIO getmode() before set: ', GPIO.getmode())
        GPIO.setmode(GPIO.BOARD)
        #print('GPIO getmode() after set: ', GPIO.getmode())

        self.motor_enable = kwargs.get('motor_enable', True)
        self.sonar_enable = kwargs.get('sonar_enable', False)
        self.TOF_enable = kwargs.get('TOF_enable', True)
        self.compass_enable = kwargs.get('compass_enable', True)
        self.MQTT_enable = kwargs.get('MQTT_enable', True)
        self.arena_mode = kwargs.get('arena_mode', True)
        self.debug_enable = kwargs.get('debug_enable', True)

        # Set up MQTT first, so it can be passed to DebugFile, for debugging purposes.
        self.comm = None
        if self.MQTT_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.comm = MQTTComm(broker_address='192.168.1.240')
            print('MQTTComm object created.')
            #self.df.writeToDebug('MQTTComm object created.')
            print('Starting MQTT loop...')
            #self.df.writeToDebug('Starting MQTT loop...')
            self.MQTT_loop_thread = threading.Thread(target=self.comm.startLoop, daemon=True)
            self.MQTT_loop_thread.start()
            print('MQTT loop started.')
            #self.df.writeToDebug('MQTT loop started.')

        self.df = DebugFile(notes='Trying with TOF sensors', mqtt_obj=self.comm, enabled=self.debug_enable)
        self.date_time_string = self.df.getDateString()

        self.df.writeToDebug('************************* In function: {}()'.format('init'))

        self.df.writeToDebug('motor enable: {}'.format(self.motor_enable))
        self.df.writeToDebug('sonar enable: {}'.format(self.sonar_enable))
        self.df.writeToDebug('TOF enable: {}'.format(self.TOF_enable))
        self.df.writeToDebug('compass enable: {}'.format(self.compass_enable))
        self.df.writeToDebug('MQTT enable: {}'.format(self.MQTT_enable))

        self.distance_meas_enable = (self.sonar_enable or self.TOF_enable)
        all_arena_meas_enabled = self.motor_enable and self.distance_meas_enable and self.MQTT_enable and self.compass_enable


        self.save_hist = kwargs.get('save_hist', False)
        self.quiet = kwargs.get('quiet', False)

        # For getting the params passed from Agent, if there are any.
        self.passed_params = {}
        check_params = []
        for param in check_params:
            if kwargs.get(param, None) is not None:
                self.passed_params[param] = kwargs.get(param, None)


        if self.arena_mode:
            assert all_arena_meas_enabled, 'Arena mode enabled, but either motor/dist/MQTT/compass not enabled.'

        if self.motor_enable:
            self.motor = Motor(left_forward_pin=32, left_reverse_pin=33, right_forward_pin=35, right_reverse_pin=37)
            print('Motor object created.')


        if self.sonar_enable:
            print('Creating sonar objects...')
            self.sonar_forward = Sonar(GPIO_TRIGGER=10, GPIO_ECHO=8)
            self.sonar_left = Sonar(GPIO_TRIGGER=24, GPIO_ECHO=22)
            self.sonar_right = Sonar(GPIO_TRIGGER=18, GPIO_ECHO=16)
            print('Sonar objects created.')


        if self.TOF_enable:
            print('Creating TOF objects...')
            self.TOF_forward = TOF(GPIO_SHUTDOWN=8, i2c_address=0x2a)
            self.TOF_left = TOF(GPIO_SHUTDOWN=22, i2c_address=0x2b)
            self.TOF_right = TOF(GPIO_SHUTDOWN=18, i2c_address=0x2c)
            self.TOF_forward.tofOpen()
            self.TOF_left.tofOpen()
            self.TOF_right.tofOpen()
            self.TOF_forward.tofStartRanging()
            self.TOF_left.tofStartRanging()
            self.TOF_right.tofStartRanging()
            #del self.TOF_forward
            #self.TOF_forward = TOF(GPIO_SHUTDOWN=8, i2c_address=0x2a)
            print('TOF objects created.')


        if self.compass_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.compass_correction_file = kwargs.get('compass_correction_file', None)

            self.compass = Compass(compass_correction_file=self.compass_correction_file, pi_max=True, flip_x=True)
            print('Compass object created.')

            # Using daemon=True will cause this thread to die when the main program dies.
            print('creating compass read loop thread...')
            self.df.writeToDebug('creating compass read loop thread...')
            self.compass_read_thread = threading.Thread(target=self.compass.readCompassLoop, kwargs={'test_time':'forever', }, daemon=True)
            print('starting compass read loop thread...')
            self.df.writeToDebug('starting compass read loop thread...')
            self.compass_read_thread.start()
            print('started.')
            self.df.writeToDebug('started.')



        if self.arena_mode:

            # All units here are in meters.
            self.wall_length = 1.25
            self.xlims = np.array([-self.wall_length/2, self.wall_length/2])
            self.ylims = np.array([-self.wall_length/2, self.wall_length/2])
            self.position = np.array([0.5*(max(self.xlims) + min(self.xlims)), 0.5*(max(self.ylims) + min(self.ylims))])
            self.bottom_corner = np.array([self.xlims[0], self.ylims[0]])
            self.lims = np.array((self.xlims,self.ylims))
            self.robot_draw_rad = self.wall_length/20.0
            self.target_draw_rad = self.robot_draw_rad

            self.dist_meas_percent_tolerance = 0.12

            self.target_positions = np.array([[.19, 0], [.55, 0], [self.wall_length, .21], [self.wall_length, .65], [.97, self.wall_length], [.58, self.wall_length], [0, 1.02], [0, .60]])
            # This makes it so the target positions are w.r.t. the origin at the center.
            self.target_positions = np.array([self.cornerOriginToCenterOrigin(pos) for pos in self.target_positions])
            #These will be the positions in meters, where (0,0) is the center of the arena.

            self.N_targets = len(self.target_positions)
            self.current_target = 0
            print('Current target is: ', self.current_target)

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
                print('test IR read: ', test_IR_read)
                assert len(test_IR_read)==self.N_targets, 'Number of targets ({}) returned from MQTTComm doesn\'t match N_targets ({}) '.format(len(test_IR_read), self.N_targets)

            else:
                # In this case, we're just gonna give the reward directly, based on its distance.
                self.reward_distance_thresh = 0.3
                self.valid_targets = np.array(list(range(8)))


            self.resetStateValues()

        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.N_actions = 4
        # This determines the order of the action indices (i.e., 0 is straight, 1 is backward, etc)
        self.action_func_list = [self.motor.goStraight, self.motor.goBackward, self.motor.turn90CCW, self.motor.turn90CW]

        self.N_state_terms = len(self.getStateVec())
        print('Robot has a state vec of length: ', self.N_state_terms)



    ########### Functions that the Agent class expects.


    def getStateVec(self):

        if self.arena_mode:
            target_pos = self.target_positions[self.current_target]

            position, compass_reading = self.getPosition()
            normed_angle = compass_reading/pi

            #return(np.concatenate((position, [normed_angle], target_pos)))
            # This will let us use the one trained with velocity, since we don't really
            # have velocity here.
            #return(np.concatenate((position, [normed_angle], [0,0], target_pos)))
            return(np.concatenate((position, [normed_angle], target_pos)))
        else:
            return(np.zeros(5))


    def getPassedParams(self):
        #This returns a dict of params that were passed to the agent, that apply to the agent.
        #So if you pass it a param for 'reward', it will return that, but it won't return the
        #default val if you didn't pass it.
        return(self.passed_params)


    def iterate(self, action):

        self.df.writeToDebug('********************* In function: iterate()')
        if self.arena_mode:
            iter_str = 'iterate() {}, elapsed time=({}). R_avg={:.4f} --> act={}. pos=({:.2f}, {:.2f}), angle={:.2f}, target={}, targets={}'.format(self.iteration, fst.getTimeDiffStr(self.init_time), self.R_tot/max(1, self.iteration), action, self.position[0], self.position[1], self.angle, self.current_target, [int(x) for x in self.getTriggerList()])
            self.df.writeToDebug(iter_str)
            self.print(iter_str)
            iter_dict = {}
            iter_dict['iteration'] = self.iteration
            iter_dict['position'] = self.position.tolist()
            iter_dict['angle'] = self.angle
            iter_dict['current_target'] = self.current_target
            iter_dict['trigger_list'] = [int(x) for x in self.getTriggerList()]
            #self.comm.publishIteration(iter_dict)

        self.doAction(action)
        self.iteration += 1
        self.df.writeToDebug('did action {}'.format(action))

        if self.arena_mode:
            self.df.writeToDebug('getting reward...')
            r = self.reward()
            self.R_tot += r
            self.df.writeToDebug('got reward {}.'.format(r))
            self.df.writeToDebug('adding to hist...')
            self.addToHist()
            self.df.writeToDebug('added. Return (r, getStateVec())')
            return(r, self.getStateVec())



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
            current_target_triggered = self.targetTriggered(self.current_target)

        if current_target_triggered:
            self.print('Current target {} reached! Calling resetTarget()'.format(self.current_target))
            self.df.writeToDebug('Current target {} reached! Calling resetTarget()'.format(self.current_target))
            self.resetTarget()
            return(1.0)
        else:
            return(-0.01)


    def initEpisode(self):
        self.df.writeToDebug('in function initEpisode()')
        self.resetTarget()
        self.resetStateValues()


    def resetStateValues(self):

        self.df.writeToDebug('in function resetStateValues()')
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


    def drawState(self, ax):

        ax.clear()
        ax.set_xlim(tuple(self.xlims))
        ax.set_ylim(tuple(self.ylims))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        puck = plt.Circle(tuple(self.position), self.robot_draw_rad, color='tomato')
        ax.add_artist(puck)

        if self.current_target is not None:
            target = plt.Circle(tuple(self.target_positions[self.current_target]), self.target_draw_rad, color='seagreen')
            ax.add_artist(target)


    def plotStateParams(self, axes):

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.clear()
        ax1.plot(self.pos_hist[:,0][-1000:],label='x')
        ax1.plot(self.pos_hist[:,1][-1000:],label='y')
        ax1.legend()

        ax2.clear()
        ax2.plot(self.action_hist[-1000:],label='a')
        ax2.set_yticks([0,1,2,3])
        ax2.set_yticklabels(['F','B','CCW','CW'])
        ax2.legend()


        ax3.clear()
        ax3.plot(self.r_hist[-1000:],label='R')
        ax3.legend()


    ########################### Functions for interacting with the environment.


    def resetTarget(self):
        other_targets = [i for i in self.valid_targets if i!=self.current_target]
        self.current_target = random.choice(other_targets)
        self.current_target_pos = self.target_positions[self.current_target]
        self.print('Target reset in resetTarget(). Current target is now {}.'.format(self.current_target))
        self.df.writeToDebug('Target reset in resetTarget(). Current target is now {}.'.format(self.current_target))


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
            self.df.writeToDebug('No mqtt reading, returning all 0.')
            return(['0']*self.N_targets)

        return(mqtt_reading['IR_reading'].split())


    def doAction(self, action):
        assert self.motor_enable, 'Motor not enabled! crashing.'
        # Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.last_action = action
        self.action_func_list[action]()


    def touchingSameWall(self, a, a_theta, b, b_theta):
        #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
        #If any are, then it returns the coord and the index of it. Otherwise, returns None.
        #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
        same_coord_acc_x = abs((x1 - x2)/self.wall_length)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        same_coord_acc_y = abs((y1 - y2)/self.wall_length)

        if same_coord_acc_x < same_coord_acc_y:
            return(x1, 0, same_coord_acc_x)
        else:
            return(y1, 1, same_coord_acc_y)


    def touchingOppWall(self, a, a_theta, b, b_theta):
        #Returns index of two that are touching opp walls, None otherwise.
        #Also returns the coordinate we're sure about now, which is the negative of the
        #negative of the pair that makes up the span.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        span_x = abs(x1 - x2)
        #print('span x={}'.format(span))
        #if abs((span - self.wall_length)/(0.5*(span + self.wall_length))) < self.dist_meas_percent_tolerance:
        span_accuracy_x = abs((span_x - self.wall_length)/self.wall_length)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        span_y = abs(y1 - y2)
        span_accuracy_y = abs((span_y - self.wall_length)/self.wall_length) # Lower is better

        if span_accuracy_x < span_accuracy_y:
            if x1 < 0:
                return(-x1, 0, span_accuracy_x)
            else:
                return(-x2, 0, span_accuracy_x)
        else:
            if y1 < 0:
                return(-y1, 1, span_accuracy_y)
            else:
                return(-y2, 1, span_accuracy_y)



    def cornerOriginToCenterOrigin(self, pos):

        # This changes it so if your coords are so that the origin is in the
        # bottom left hand corner, now it's in the middle of the arena.

        center_origin_pos = pos + self.bottom_corner
        return(center_origin_pos)


    def calculatePosition(self, d1, d2, d3, theta):

        self.df.writeToDebug('************************* In function: {}()'.format('calculatePosition'))
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
        pair12 = [d1, theta, d2, theta + pi/2]
        pair23 = [d2, theta + pi/2, d3, theta - pi/2]
        pair13 = [d1, theta, d3, theta - pi/2]

        pair12_same = self.touchingSameWall(*pair12)
         # 2 and 3 should never be able to hit the same wall... and it makes trouble when they hit opposite walls at the same height!
        #pair23_same = self.touchingSameWall(*pair23)
        pair23_same = None
        pair13_same = self.touchingSameWall(*pair13)


        pair12_opp = self.touchingOppWall(*pair12)
        pair23_opp = self.touchingOppWall(*pair23)
        pair13_opp = self.touchingOppWall(*pair13)

        self.df.writeToDebug('Same walls: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))
        self.df.writeToDebug('Opp walls: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(pair12_opp, pair23_opp, pair13_opp))

        same_accs = [('12', pair12_same), ('13', pair13_same)]
        opp_accs = [('12', pair12_opp), ('23', pair23_opp), ('13', pair13_opp)]

        # This should sort for the lowest accuracy returned in the tuple.
        best_acc_tuple_same = sorted(same_accs, key=lambda x: x[1][2])[0]
        best_acc_tuple_opp = sorted(opp_accs, key=lambda x: x[1][2])[0]

        best_acc_same = best_acc_tuple_same[1][2]
        best_acc_opp = best_acc_tuple_opp[1][2]

        self.df.writeToDebug('Best accuracy, same wall: {}'.format(best_acc_tuple_same))
        self.df.writeToDebug('Best accuracy, opp wall: {}'.format(best_acc_tuple_opp))

        sol = np.array([0.0, 0.0])


        best_acc_percent_diff = abs(best_acc_same - best_acc_opp)

        if best_acc_percent_diff < self.dist_meas_percent_tolerance:
            same, opp = True, True
            self.df.writeToDebug('Percent diff between best accs, {:.3f}, is smaller than tolerance {:.3f}, touching same AND opp walls'.format(best_acc_percent_diff, self.dist_meas_percent_tolerance))
        else:
            if best_acc_same < best_acc_opp:
                same, opp = True, False
                self.df.writeToDebug('Best same acc is better than best opp acc, touching only same wall')
            else:
                same, opp = False, True
                self.df.writeToDebug('Best opp acc is better than best same acc, touching only opp wall')


        if (same and not opp) or (opp and not same):
            if same and not opp:
                #self.df.writeToDebug('Touching same wall: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))

                dist, coord, acc = best_acc_tuple_same[1]

                if best_acc_tuple_same[0] == '12':
                    other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if best_acc_tuple_same[0] == '23':
                    other_ray = [d1*cos(theta), d1*sin(theta)]

                if best_acc_tuple_same[0] == '13':
                    other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

                #Sets the coordinate we've figured out.
                if dist>=0:
                    sol[coord] = self.wall_length - dist
                else:
                    sol[coord] = -dist

            if opp and not same:
                #This means that no two touch the same wall.
                #print('opp walls, not the same')
                #self.df.writeToDebug('Touching opp wall: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(pair12_opp, pair23_opp, pair13_opp))

                dist, coord, acc = best_acc_tuple_opp[1]

                if best_acc_tuple_opp[0] == '12':
                    other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if best_acc_tuple_opp[0] == '23':
                    other_ray = [d1*cos(theta), d1*sin(theta)]

                if best_acc_tuple_opp[0] == '13':
                    other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

                #The dist should already be positive.
                sol[coord] = dist

            #This is the other coord we don't have yet, which works for either case.
            other_coord = abs(1 - coord)
            other_dist = other_ray[other_coord]

            self.df.writeToDebug('dist={:.3f}, coord={}, other_coord={}, other_ray=({:.3f}, {:.3f})'.format(dist, coord, other_coord, other_ray[0], other_ray[1]))

            if other_dist>=0:
                sol[other_coord] = self.wall_length - other_dist
            else:
                sol[other_coord] = -other_dist

            pos = self.cornerOriginToCenterOrigin(sol)
            self.df.writeToDebug('pos. calcd in calcPosition=({:.3f}, {:.3f})'.format(pos[0], pos[1]))
            return(pos)


        if same and opp:
            #print('unsolvable case, touching same wall and spanning. Attempting best guess')
            #self.df.writeToDebug('Touching same AND opp walls.')

            dist, coord, acc = best_acc_tuple_same[1]

            if best_acc_tuple_same[0] == '12':
                other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

            if best_acc_tuple_same[0] == '23':
                other_ray = [d1*cos(theta), d1*sin(theta)]

            if best_acc_tuple_same[0] == '13':
                other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

            #Sets the coordinate we've figured out.
            if dist>=0:
                sol[coord] = self.wall_length - dist
            else:
                sol[coord] = -dist

            #That was for the coord you can find out for sure. Now we try to get a good estimate
            #for the other coord by taking the average of what it could be at the extremes.

            #This is the other coord we don't have yet.
            other_coord = abs(1 - coord)

            other_coord_vals = [[d1*cos(theta), d1*sin(theta)][other_coord],
                                [d2*cos(theta + pi/2), d2*sin(theta + pi/2)][other_coord],
                                [d3*cos(theta - pi/2), d3*sin(theta - pi/2)][other_coord]]

            #these are how far below and above x and y the rays are.
            #I think below_margin HAS to be negative, and above has to be positive.
            below_margin = min(other_coord_vals)
            above_margin = max(other_coord_vals)

            sol[other_coord] = (-below_margin + (self.wall_length - above_margin + below_margin)/2.0)
            pos = self.cornerOriginToCenterOrigin(sol)
            self.df.writeToDebug('pos. calcd in calcPosition=({:.3f}, {:.3f})'.format(pos[0], pos[1]))
            return(pos)


        # This is if something is wrong and it can't figure out the position.
        self.df.writeToDebug('Couldnt calc position based on meas., returning default val of (0,0) (in center origin coords).')
        return(sol)


    def getPosition(self):

        self.df.writeToDebug('************************* In function: {}()'.format('getPosition'))

        assert self.distance_meas_enable, 'No distance sensors enabled! exiting.'
        assert self.compass_enable, 'No compass! exiting.'

        d1, d2, d3 = self.getDistanceMeas()

        compass_reading = self.getCompassDirection() #From now on, the function will prepare and scale everything.

        self.df.writeToDebug('Raw measurements: d1={:.3f}, d2={:.3f}, d3={:.3f}, angle={:.3f}'.format(d1, d2, d3, compass_reading))

        self.position = self.calculatePosition(d1, d2, d3, compass_reading)
        self.angle = compass_reading

        return(self.position, compass_reading)


    def getCompassDirection(self):

        assert self.compass_enable, 'Compass not enabled in getCompassDirection()!'

        return(self.compass.getCompassDirection())


    def getDistanceMeas(self):

        # This is for interfacing to whatever distance measuring thing you're using, sonar or TOF.
        # It only gets the dist. measures, so you can run it not in arena_mode.

        assert self.distance_meas_enable, 'No distance sensors enabled! exiting.'

        if self.sonar_enable:
            return(self.getSonarMeas())

        if self.TOF_enable:
            return(self.getTOFMeas())


    def getSonarMeas(self):

        assert self.sonar_enable, 'Trying to cal getSonarMeas() but sonar not enabled!'

        i = 0
        attempts = 5
        delay = 0.01
        while i<attempts:
            d1 = self.sonar_forward.distance()
            time.sleep(delay)
            d2 = self.sonar_left.distance()
            time.sleep(delay)
            d3 = self.sonar_right.distance()
            time.sleep(delay)

            # 1.5 being a slight overestimation of sqrt(2) here, the max distance
            # it should be able to measure.
            if (d1 > self.wall_length*1.5) or (d2 > self.wall_length*1.5) or (d3 > self.wall_length*1.5):
                i += 1
                self.df.writeToDebug('Sonar meas. attempt {} failed: d1={:.3f}, d2={:.3f}, d3={:.3f}, retrying'.format(i, d1, d2, d3))
            else:
                return(d1, d2, d3)

        d1 = self.sonar_forward.distance()
        time.sleep(delay)
        d2 = self.sonar_left.distance()
        time.sleep(delay)
        d3 = self.sonar_right.distance()
        time.sleep(delay)
        max_dist = self.wall_length*1.5
        return(min(d1, max_dist), min(d2, max_dist), min(d3, max_dist))


    def getTOFMeas(self):

        time.sleep(0.1)
        front_sensor_offset = 0.13
        left_sensor_offset = 0.03
        right_sensor_offset = 0.03
        d1 = self.TOF_forward.distance() + front_sensor_offset
        d2 = self.TOF_left.distance() + left_sensor_offset
        d3 = self.TOF_right.distance() + right_sensor_offset

        if self.arena_mode:
            max_dist = self.wall_length*1.4
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
            print('testing motor!')
            #self.motor.wheelTest(test_time=2)

        if self.sonar_enable:
            print('testing sonar! (front)')
            self.sonar_forward.distanceTestLoop(test_time=test_duration)
            print('testing sonar! (left)')
            self.sonar_left.distanceTestLoop(test_time=test_duration)
            print('testing sonar! (right)')
            self.sonar_right.distanceTestLoop(test_time=test_duration)

        if self.TOF_enable:
            print('testing TOF! (front)')
            self.TOF_forward.distanceTestLoop(test_time=test_duration)
            print('testing TOF! (left)')
            self.TOF_left.distanceTestLoop(test_time=test_duration)
            print('testing TOF! (right)')
            self.TOF_right.distanceTestLoop(test_time=test_duration)

        if self.compass_enable:
            print('testing compass!')
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            self.compass.readCompassLoop(test_time=test_duration, print_readings=True)


    def DCloop(self, stdscr):
        #https://docs.python.org/3/howto/curses.html
        #https://docs.python.org/3/library/curses.html#curses.window.clrtobot
        self.df.writeToDebug('************************* In function: {}()'.format('DCloop'))
        self.df.writeToDebug('Size of curses window: LINES={}, COLS={}'.format(curses.LINES, curses.COLS))
        delay_time = 0.1

        move_str_pos = [0, 6]

        self.drawStandard(stdscr)

        while True:
            c = stdscr.getch()

            if c == curses.KEY_LEFT:
                self.df.writeToDebug('Pressed Left key, turning CCW')
                self.iterate(2)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Left key, turning CCW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_RIGHT:
                self.df.writeToDebug('Pressed Right key, turning CW')
                self.iterate(3)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Right key, turning CW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_UP:
                self.df.writeToDebug('Pressed Up key, going straight')
                self.iterate(0)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Up key, going straight')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_DOWN:
                self.df.writeToDebug('Pressed Down key, going backwards')
                self.iterate(1)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Down key, going backwards')
                self.moveCursorRefresh(stdscr)


            if c == ord('r'):
                self.df.writeToDebug('Pressed r, refreshing')
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed r, refreshing')
                self.moveCursorRefresh(stdscr)


            elif c == ord('q'):
                print('you pressed q! exiting')
                break  # Exit the while loop


    def moveCursorRefresh(self, stdscr):
        stdscr.move(curses.LINES - 1, curses.COLS - 1)
        stdscr.refresh() #Do this after addstr


    def directControl(self):
        print('entering curses loop')
        curses.wrapper(self.DCloop)
        print('exited curses loop.')



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
            self.df.writeToDebug('Position PERCENT of box {} out of bounds in redrawBox(), setting to 0.5,0.5 for drawing.'.format(pos))
            pos = [0.5, 0.5]
        stdscr.addstr(box_coord_y + -1 + (box_side_y - int(pos[1]*box_side_y)), box_coord_x + 1 + int(pos[0]*2*box_side_x), arrow_list[angle_ind])


    def drawStandard(self, stdscr):
        stdscr.erase()
        self.df.writeToDebug('************************* In function: {}()'.format('drawStandard'))

        if self.distance_meas_enable:
            self.df.writeToDebug('Getting distance info')
            d1, d2, d3 = self.getDistanceMeas()
            info_str = 'Distance meas: (straight = {:.2f}, left = {:.2f}, right = {:.2f})'.format(d1, d2, d3)
            self.df.writeToDebug(info_str)
            stdscr.addstr(0, 0,  info_str)

        if self.compass_enable:
            self.df.writeToDebug('Getting compass info')
            compass_reading = self.getCompassDirection() #From now on, the function will prepare and scale everything.
            info_str = 'Compass meas: ({:.2f})'.format(compass_reading)
            self.df.writeToDebug(info_str)
            stdscr.addstr(2, 0,  info_str)


        if self.arena_mode:

            self.df.writeToDebug('Getting position info')

            #Draw box, with best position guess
            box_side_x = 25
            box_side_y = box_side_x
            box_coord_y = 9
            box_coord_x = 45

            # getPosition should return the position, where (0,0) is the center of
            # the arena. box_pos is the integer pair for the drawn box indices.

            try:
                raw_pos, angle = self.getPosition()
                pos_str = '({:.2f}, {:.2f})'.format(raw_pos[0], raw_pos[1])
                self.df.writeToDebug('Got position info: {}'.format(pos_str))
                stdscr.addstr(4, 0, 'Estimated position: {}'.format(pos_str))

                box_percent_pos = (raw_pos - self.bottom_corner)/self.wall_length

            except:
                box_percent_pos = np.array([0.5, 0.5])
                angle = 0

            self.df.writeToDebug('drawing agent pos (o) at percent ({:.3f}, {:.3f})'.format(box_percent_pos[0], box_percent_pos[1]))
            box_info = (box_coord_y, box_coord_x, box_side_y, box_side_x)
            self.redrawBox(stdscr, box_info, box_percent_pos, angle)


        if self.MQTT_enable:
            self.df.writeToDebug('Getting IR target info...')
            reading_str = ' '.join(self.pollTargetServer())
            IR_read = 'IR target reading:  ' + reading_str
            self.df.writeToDebug('Got IR target info: {}'.format(reading_str))
            stdscr.addstr(8, 0,  IR_read)

        stdscr.addstr(curses.LINES - 1, 0,  'Press q or Esc to quit')

        stdscr.refresh() #Do this after addstr
        self.df.writeToDebug('end of drawStandard()\n\n')


    ############ Bookkeeping functions

    def print(self, print_str):
        if not self.quiet:
            print(print_str)


    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.position]))
        self.angle_hist = np.concatenate((self.angle_hist, [self.angle]))
        self.iterations = np.concatenate((self.iterations, [self.iteration]))
        self.t = np.concatenate((self.t, [time.time() - self.start_time]))
        self.r_hist = np.concatenate((self.r_hist, [self.reward()]))
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

        print('allhist shape: ', all_hist.shape)
        header = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('iter', 't', 'x', 'y', 'ang', 'r', 'action', 'target')

        default_fname = 'all_hist_' + self.date_time_string + '.txt'
        fname = kwargs.get('fname', default_fname)
        print('\nSaving robot hist to', fname)
        np.savetxt(fname, all_hist, header=header, fmt='%.3f', delimiter='\t')



    def __del__(self):

        del self.motor
        time.sleep(0.5)
        if self.TOF_enable:
            print('\n\nDeleting TOF objects...')
            del self.TOF_forward
            del self.TOF_left
            del self.TOF_right
        time.sleep(0.5)
        GPIO.cleanup()
