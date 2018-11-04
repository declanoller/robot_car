calculatePositionimport time
import RPi.GPIO as GPIO
from Motor import Motor
from Sonar import Sonar
from Compass import Compass
from CommMQTT import CommMQTT

import matplotlib.pyplot as plt
from random import randint, random, sample, choice
import numpy as np
from math import sin, cos, tan, pi

import curses


class Robot:

    def __init__(self, motor_enable=True, sonar_enable=True, compass_enable=True, **kwargs):


        #GPIO Mode (BOARD / BCM)
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)


        self.motor_enable = motor_enable
        self.sonar_enable = sonar_enable
        self.compass_enable = compass_enable

        if self.motor_enable:
            self.motor = Motor(left_forward_pin=31, left_reverse_pin=33, right_forward_pin=35, right_reverse_pin=37)
            print('Motor object created.')
        if self.sonar_enable:
            self.sonar_forward = Sonar(GPIO_TRIGGER=10, GPIO_ECHO=8)
            self.sonar_left = Sonar(GPIO_TRIGGER=24, GPIO_ECHO=22)
            self.sonar_right = Sonar(GPIO_TRIGGER=18, GPIO_ECHO=16)
            print('Sonar objects created.')
        if self.compass_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.compass = Compass()
            print('Compass object created.')
        if self.MQTT_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.comm = CommMQTT()
            print('CommMQTT object created.')


        self.N_state_terms = 6

        self.passed_params = {}
        check_params = []
        for param in check_params:
            if kwargs.get(param, None) is not None:
                self.passed_params[param] = kwargs.get(param, None)

        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.N_actions = 4
        self.arena_side = 1.0
        self.xlims = np.array([-0.5,0.5])
        self.ylims = np.array([-0.5,0.5])
        self.lims = np.array((self.xlims,self.ylims))
        self.robot_draw_rad = self.arena_side/20.0
        self.target_draw_rad = self.robot_draw_rad

        #These will be the positions in meters, where (0,0) is the center of the arena.
        self.target_positions = [(-0.5, -0.5), (0.5, 0.5)]
        self.N_targets = len(self.target_positions)

        test_IR_read = self.comm.getLatestReadingIR()
        assert len(test_IR_read)==self.N_targets, 'Number of targets ({}) returned from CommMQTT doesn\'t match N_targets ({}) '.format(len(test_IR_read), self.N_targets)
        self.current_target = 0



    ########### Functions that the Agent class expects.


    def getStateVec(self):

        target_pos = self.target_positions[self.current_target]

        position, compass_reading = self.getPosition()
        normed_angle = compass_reading/pi

        return(np.concatenate((position, normed_angle, target_pos)))


    def getPassedParams(self):
        #This returns a dict of params that were passed to the agent, that apply to the agent.
        #So if you pass it a param for 'reward', it will return that, but it won't return the
        #default val if you didn't pass it.
        return(self.passed_params)


    def iterate(self, action):
        self.doAction(action)

        r = self.reward()
        if r > 0:
            self.resetTarget()

        return(r, self.getStateVec())


    def reward(self):
        triggered_IR_sensors = self.pollTargetServer()
        if triggered_IR_sensors[self.current_target]==1:
            self.resetTarget()
            return(1.0)
        else:
            return(-0.01)


    def initEpisode(self):
        self.resetStateValues()
        self.resetTarget()


    def resetStateValues(self):

        self.pos, self.angle = self.getPosition()

        self.pos_hist = np.array([self.pos])
        self.angle_hist = np.array([self.angle])
        self.action_hist = [0]
        self.t = [0]
        self.r_hist = []


    def drawState(self, ax):

        ax.clear()
        ax.set_xlim(tuple(self.xlims))
        ax.set_ylim(tuple(self.ylims))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        puck = plt.Circle(tuple(self.pos), self.robot_draw_rad, color='tomato')
        ax.add_artist(puck)

        if self.target is not None:
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
        other_sensors = [i for i in range(self.N_targets) if i!=self.current_target]
        self.current_target = random.choice(other_sensors)


    def pollTargetServer(self):
        return(self.comm.getLatestReadingIR())


    def doAction(self, action):

        assert self.motor_enable, 'Motor not enabled! crashing.'

        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        if action==0:
            self.motor.goStraight()
        elif action==1:
            self.motor.goBackward()
        elif action==2:
            self.motor.turn90CCW()
        elif action==3:
            self.motor.turn90CW()


    def touchingSameWall(self, a, a_theta, b, b_theta):
        #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
        #If any are, then it returns the coord and the index of it. Otherwise, returns None.
        #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.
        percent_tolerance = 0.02

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
        if abs((x1 - x2)/(0.5*(x1 + x2))) < percent_tolerance:
            return(x1, 0)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        if abs((y1 - y2)/(0.5*(y1 + y2))) < percent_tolerance:
            return(y1, 1)

        return(None)


    def touchingOppWall(self, a, a_theta, b, b_theta):
        #Returns index of two that are touching opp walls, None otherwise.
        #Also returns the coordinate we're sure about now, which is the negative of the
        #negative of the pair that makes up the span.
        percent_tolerance = 0.02

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        span = abs(x1 - x2)
        wall_dist = 1.0
        #print('span x={}'.format(span))
        if abs((span - wall_dist)/(0.5*(span + wall_dist))) < percent_tolerance:
            if x1 < 0:
                return(-x1, 0)
            else:
                return(-x2, 0)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        span = abs(y1 - y2)
        #print('span y={}'.format(span))
        wall_dist = 1.0
        if abs((span - wall_dist)/(0.5*(span + wall_dist))) < percent_tolerance:
            if y1 < 0:
                return(-y1, 1)
            else:
                return(-y2, 1)

        return(None)


    def calculatePosition(self, d1, d2, d3, theta):

        #This uses some...possibly sketchy geometry, but I think it should work
        #generally, no matter which direction it's pointed in.
        #
        #There are 3 possibilities for the configuration: two sonars are hitting the same wall,
        #two sonars are hitting opposite walls, or both.
        #If it's one of the first two, the position is uniquely specified, and you just have to
        #do the painful geometry for it. If it's the third, it's actually not specified, and you
        #can only make an educated guess within some range.
        #
        #d1 is the front sonar, d2 the left, d3 the right.
        pair12 = [d1, theta, d2, theta + pi/2]
        pair23 = [d2, theta + pi/2, d3, theta - pi/2]
        pair13 = [d1, theta, d3, theta - pi/2]

        pair12_same = touchingSameWall(*pair12)
        pair23_same = touchingSameWall(*pair23)
        pair13_same = touchingSameWall(*pair13)

        pair12_opp = touchingOppWall(*pair12)
        pair23_opp = touchingOppWall(*pair23)
        pair13_opp = touchingOppWall(*pair13)

        sol = {}
        same = (pair12_same is not None) or (pair23_same is not None) or (pair13_same is not None)
        opp = (pair12_opp is not None) or (pair23_opp is not None) or (pair13_opp is not None)

        if (same and not opp) or (opp and not same):
            if same and not opp:
                print('two touching same wall')

                if pair12_same is not None:
                    dist, coord = pair12_same
                    other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if pair23_same is not None:
                    dist, coord = pair23_same
                    other_ray = [d1*cos(theta), d1*sin(theta)]

                if pair13_same is not None:
                    dist, coord = pair13_same
                    other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

                #Sets the coordinate we've figured out.
                if dist>=0:
                    sol[coord] = 1.0 - dist
                else:
                    sol[coord] = -dist

            if opp and not same:
                #This means that no two touch the same wall.
                print('opp walls, not the same')

                if pair12_opp is not None:
                    dist, coord = pair12_opp
                    other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if pair23_opp is not None:
                    dist, coord = pair23_opp
                    other_ray = [d1*cos(theta), d1*sin(theta)]

                if pair13_opp is not None:
                    dist, coord = pair13_opp
                    other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

                #The dist should already be positive.
                sol[coord] = dist

            #This is the other coord we don't have yet.
            other_coord = abs(1 - coord)
            other_dist = other_ray[other_coord]

            if other_dist>=0:
                sol[other_coord] = 1.0 - other_dist
            else:
                sol[other_coord] = -other_dist

            return(sol[0], sol[1])


        if same and opp:
            print('unsolvable case, touching same wall and spanning. Attempting best guess')

            if pair12_same is not None:
                dist, coord = pair12_same
                other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

            if pair23_same is not None:
                dist, coord = pair23_same
                other_ray = [d1*cos(theta), d1*sin(theta)]

            if pair13_same is not None:
                dist, coord = pair13_same
                other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

            #Sets the coordinate we've figured out.
            if dist>=0:
                sol[coord] = 1.0 - dist
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

            sol[other_coord] = (-below_margin + (1 - above_margin))/2.0
            return(sol[0], sol[1])


    def getPosition(self):

        assert self.sonar_enable, 'No sonar! exiting.'
        assert self.compass_enable, 'No compass! exiting.'

        d1 = self.sonar_forward.distance()
        d2 = self.sonar_left.distance()
        d3 = self.sonar_right.distance()
        compass_reading = self.compass.getCompassDirection() #From now on, the function will prepare and scale everything.

        position = self.calculatePosition(d1, d2, d3, compass_reading)
        self.position = position
        self.angle = compass_reading

        return(position, compass_reading)

    ################# Functions for interacting directly with the robot.


    def testAllDevices(self):

        if self.motor_enable:
            print('testing motor!')
            #self.motor.wheelTest(test_time=2)
        if self.sonar_enable:
            print('testing sonar! (front)')
            self.sonar_forward.distanceTestLoop(test_time=3)
            print('testing sonar! (left)')
            self.sonar_left.distanceTestLoop(test_time=3)
            print('testing sonar! (right)')
            self.sonar_right.distanceTestLoop(test_time=3)
        if self.compass_enable:
            print('testing compass!')
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            self.compass.readCompassLoop(test_time=4)


    def DCloop(self, stdscr):

        delay_time = 0.2

        while True:
            c = stdscr.getch()
            if c == curses.KEY_LEFT:
                stdscr.addstr('Pressed Left key, turning CCW\n')
                self.doAction(2)
                time.sleep(delay_time)

            if c == curses.KEY_RIGHT:
                stdscr.addstr('Pressed Right key, turning CW\n')
                self.doAction(3)
                time.sleep(delay_time)

            if c == curses.KEY_UP:
                stdscr.addstr('Pressed Up key, going straight\n')
                self.doAction(0)
                time.sleep(delay_time)

            if c == curses.KEY_DOWN:
                stdscr.addstr('Pressed Down key, going backwards\n')
                self.doAction(1)
                time.sleep(delay_time)

            elif c == ord('q'):
                print('you pressed q! exiting')
                break  # Exit the while loop


    def directControl(self):
        print('entering curses loop')
        curses.wrapper(self.DCloop)
        print('exited curses loop.')





    ############ Bookkeeping functions

    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.position]))
        self.angle_hist = np.concatenate((self.angle_hist, [self.angle]))
        self.t.append(self.t[-1] + self.time_step)
        self.r_hist.append(self.reward())


    def __del__(self):
        del self.motor
        GPIO.cleanup()
