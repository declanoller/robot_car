import time
import RPi.GPIO as GPIO
from Motor import Motor
from Sonar import Sonar
from Compass import Compass
from CommMQTT import CommMQTT
from DebugFile import DebugFile

import matplotlib.pyplot as plt
from random import randint, random, sample, choice
import numpy as np
from math import sin, cos, tan, pi

import threading
import curses


class Robot:

    def __init__(self, motor_enable=True, sonar_enable=True, compass_enable=True, MQTT_enable=True, **kwargs):


        #GPIO Mode (BOARD / BCM)
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)

        self.df = DebugFile(notes='figuring out freezing problem')

        self.df.writeToDebug('************************* In function: {}()'.format('init'))

        self.motor_enable = motor_enable
        self.sonar_enable = sonar_enable
        self.compass_enable = compass_enable
        self.MQTT_enable = MQTT_enable

        self.df.writeToDebug('motor enable: {}'.format(self.motor_enable))
        self.df.writeToDebug('sonar enable: {}'.format(self.sonar_enable))
        self.df.writeToDebug('compass enable: {}'.format(self.compass_enable))
        self.df.writeToDebug('MQTT enable: {}'.format(self.MQTT_enable))

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
            compass_correction = {}
            compass_correction['ideal_angles'] = np.array([pi, pi*3/4, pi*2/4, pi*1/4, pi*0/4, -pi*1/4, -pi*2/4, -pi*3/4, -pi])
            compass_correction['meas_angles'] = np.array([pi, 2.47, 1.85, 1.16, 0.35, -0.57, -1.69, -2.46, -pi])

            self.compass = Compass(compass_correction)
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




        if self.MQTT_enable:
            #Compass uses I2C pins, which are 3 (SDA) and 5 (SCL) for the RPi 3.
            self.comm = CommMQTT(broker_address='192.168.1.240')
            print('CommMQTT object created.')
            self.df.writeToDebug('CommMQTT object created.')


        self.N_state_terms = 6

        self.passed_params = {}
        check_params = []
        for param in check_params:
            if kwargs.get(param, None) is not None:
                self.passed_params[param] = kwargs.get(param, None)

        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.N_actions = 4
        self.wall_length = 1.25
        self.xlims = np.array([-self.wall_length/2, self.wall_length/2])
        self.ylims = np.array([-self.wall_length/2, self.wall_length/2])
        self.bottom_corner = np.array([self.xlims[0], self.ylims[0]])
        self.lims = np.array((self.xlims,self.ylims))
        self.robot_draw_rad = self.wall_length/20.0
        self.target_draw_rad = self.robot_draw_rad

        self.dist_meas_percent_tolerance = 0.2

        #These will be the positions in meters, where (0,0) is the center of the arena.

        #This is in terms of x and y, from the bottom left corner.
        self.target_positions = np.array([[.19, 0], [.55, 0], [self.wall_length, .21], [self.wall_length, .65], [.97, self.wall_length], [.58, self.wall_length], [0, 1.02], [0, .60]])
        self.N_targets = len(self.target_positions)

        if self.MQTT_enable:
            time.sleep(0.2)
            test_IR_read = self.pollTargetServer()
            print('test IR read: ', test_IR_read)
            assert len(test_IR_read)==self.N_targets, 'Number of targets ({}) returned from CommMQTT doesn\'t match N_targets ({}) '.format(len(test_IR_read), self.N_targets)

            self.current_target = 0

        # This determines the order of the action indices (i.e., 0 is straight, 1 is backward, etc)
        self.action_func_list = [self.motor.goStraight, self.motor.goBackward, self.motor.turn90CCW, self.motor.turn90CW]



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
        self.current_target_pos = self.target_positions[self.current_target]


    def pollTargetServer(self):
        return(self.comm.getLatestReadingIR()['IR_reading'].split())


    def doAction(self, action):
        assert self.motor_enable, 'Motor not enabled! crashing.'
        # Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.action_func_list[action]()


    def touchingSameWall(self, a, a_theta, b, b_theta):
        #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
        #If any are, then it returns the coord and the index of it. Otherwise, returns None.
        #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
        if abs((x1 - x2)/(0.5*(x1 + x2))) < self.dist_meas_percent_tolerance:
            return(x1, 0)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        if abs((y1 - y2)/(0.5*(y1 + y2))) < self.dist_meas_percent_tolerance:
            return(y1, 1)

        return(None)


    def touchingOppWall(self, a, a_theta, b, b_theta):
        #Returns index of two that are touching opp walls, None otherwise.
        #Also returns the coordinate we're sure about now, which is the negative of the
        #negative of the pair that makes up the span.

        #x
        x1 = a*cos(a_theta)
        x2 = b*cos(b_theta)
        span = abs(x1 - x2)
        #print('span x={}'.format(span))
        #if abs((span - self.wall_length)/(0.5*(span + self.wall_length))) < self.dist_meas_percent_tolerance:
        span_accuracy = abs((span - self.wall_length)/self.wall_length)
        if span_accuracy < self.dist_meas_percent_tolerance:
            # This current way of doing it anchors it to the LEFT/DOWN, so it will always have that bias.
            # Instead let's try taking the average.
            if x1 < 0:
                return(-x1, 0, span_accuracy)
            else:
                return(-x2, 0, span_accuracy)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        span = abs(y1 - y2)
        span_accuracy = abs((span - self.wall_length)/self.wall_length) # Lower is better
        if span_accuracy < self.dist_meas_percent_tolerance:
            if y1 < 0:
                return(-y1, 1, span_accuracy)
            else:
                return(-y2, 1, span_accuracy)

        return(None)


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
        # Here, it will return the coords where the origin is the bottom left
        # corner, so it spans x=(0, wall_length), y=(0, wall_length).
        #
        #
        pair12 = [d1, theta, d2, theta + pi/2]
        pair23 = [d2, theta + pi/2, d3, theta - pi/2]
        pair13 = [d1, theta, d3, theta - pi/2]

        pair12_same = self.touchingSameWall(*pair12)
        pair23_same = self.touchingSameWall(*pair23)
        pair13_same = self.touchingSameWall(*pair13)

        pair12_opp = self.touchingOppWall(*pair12)
        pair23_opp = self.touchingOppWall(*pair23)
        pair13_opp = self.touchingOppWall(*pair13)

        sol = np.array([0.0, 0.0])
        same = (pair12_same is not None) or (pair23_same is not None) or (pair13_same is not None)
        opp = (pair12_opp is not None) or (pair23_opp is not None) or (pair13_opp is not None)

        if (same and not opp) or (opp and not same):
            if same and not opp:
                #print('two touching same wall')
                self.df.writeToDebug('Touching same wall: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))

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
                    sol[coord] = self.wall_length - dist
                else:
                    sol[coord] = -dist

            if opp and not same:
                #This means that no two touch the same wall.
                #print('opp walls, not the same')
                self.df.writeToDebug('Touching opp wall: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(pair12_opp, pair23_opp, pair13_opp))

                best_span_acc = 1.0 # Lower is better for this.

                if pair12_opp is not None:
                    temp_dist, temp_coord, span_accuracy = pair12_opp
                    if span_accuracy <= best_span_acc:
                        dist = temp_dist
                        coord = temp_coord
                        best_span_acc = span_accuracy
                        other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if pair23_opp is not None:
                    temp_dist, temp_coord, span_accuracy = pair23_opp
                    if span_accuracy <= best_span_acc:
                        dist = temp_dist
                        coord = temp_coord
                        best_span_acc = span_accuracy
                        other_ray = [d1*cos(theta), d1*sin(theta)]

                if pair13_opp is not None:
                    temp_dist, temp_coord, span_accuracy = pair13_opp
                    if span_accuracy <= best_span_acc:
                        dist = temp_dist
                        coord = temp_coord
                        best_span_acc = span_accuracy
                        other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

                #The dist should already be positive.
                sol[coord] = dist

            #This is the other coord we don't have yet.
            other_coord = abs(1 - coord)
            other_dist = other_ray[other_coord]

            self.df.writeToDebug('dist={:.3f}, coord={}, other_coord={}, other_ray=({:.3f}, {:.3f})'.format(dist, coord, other_coord, other_ray[0], other_ray[1]))

            if other_dist>=0:
                sol[other_coord] = self.wall_length - other_dist
            else:
                sol[other_coord] = -other_dist

            return(self.cornerOriginToCenterOrigin(sol))

        if same and opp:
            #print('unsolvable case, touching same wall and spanning. Attempting best guess')
            self.df.writeToDebug('Touching same AND opp walls.')

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
            return(self.cornerOriginToCenterOrigin(sol))


        # This is if something is wrong and it can't figure out the position.
        self.df.writeToDebug('Couldnt calc position based on meas., returning default val of (0,0) (in center origin coords).')
        return(sol)


    def getPosition(self):

        self.df.writeToDebug('************************* In function: {}()'.format('getPosition'))

        assert self.sonar_enable, 'No sonar! exiting.'
        assert self.compass_enable, 'No compass! exiting.'

        d1, d2, d3 = self.getSonarMeas()

        compass_reading = self.compass.getCompassDirection() #From now on, the function will prepare and scale everything.

        self.df.writeToDebug('Raw measurements: d1={:.3f}, d2={:.3f}, d3={:.3f}, angle={:.3f}'.format(d1, d2, d3, compass_reading))

        self.position = self.calculatePosition(d1, d2, d3, compass_reading)
        self.angle = compass_reading

        return(self.position, compass_reading)


    def getSonarMeas(self):

        assert self.sonar_enable, 'Trying to cal getSonarMeas() but sonar not enabled!'

        i = 0
        attempts = 5
        delay = 0.05
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
        return(min(d1, self.wall_length*1.5), min(d2, self.wall_length*1.5), min(d3, self.wall_length*1.5))


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
        #https://docs.python.org/3/howto/curses.html
        #https://docs.python.org/3/library/curses.html#curses.window.clrtobot
        self.df.writeToDebug('************************* In function: {}()'.format('DCloop'))
        self.df.writeToDebug('Size of curses window: LINES={}, COLS={}'.format(curses.LINES, curses.COLS))
        delay_time = 0.6

        move_str_pos = [0, 6]

        self.drawStandard(stdscr)

        while True:
            c = stdscr.getch()

            if c == curses.KEY_LEFT:
                self.df.writeToDebug('Pressed Left key, turning CCW')
                self.doAction(2)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Left key, turning CCW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_RIGHT:
                self.df.writeToDebug('Pressed Right key, turning CW')
                self.doAction(3)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Right key, turning CW')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_UP:
                self.df.writeToDebug('Pressed Up key, going straight')
                self.doAction(0)
                time.sleep(delay_time)
                self.drawStandard(stdscr)
                stdscr.addstr(move_str_pos[1], move_str_pos[0], 'Pressed Up key, going straight')
                self.moveCursorRefresh(stdscr)


            if c == curses.KEY_DOWN:
                self.df.writeToDebug('Pressed Down key, going backwards')
                self.doAction(1)
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
        stdscr.addstr(box_coord_y + -1 + (box_side_y - int(pos[1]*box_side_y)), box_coord_x + 1 + int(pos[0]*2*box_side_x), arrow_list[angle_ind])


    def drawStandard(self, stdscr):
        stdscr.erase()
        self.df.writeToDebug('************************* In function: {}()'.format('drawStandard'))

        if self.sonar_enable:
            self.df.writeToDebug('Getting sonar info')
            d1, d2, d3 = self.getSonarMeas()
            info_str = 'Sonar meas: (straight = {:.2f}, left = {:.2f}, right = {:.2f})'.format(d1, d2, d3)
            self.df.writeToDebug(info_str)
            stdscr.addstr(0, 0,  info_str)

        if self.compass_enable:
            self.df.writeToDebug('Getting compass info')
            compass_reading = self.compass.getCompassDirection() #From now on, the function will prepare and scale everything.
            info_str = 'Compass meas: ({:.2f})'.format(compass_reading)
            self.df.writeToDebug(info_str)
            stdscr.addstr(2, 0,  info_str)


        if self.sonar_enable and self.compass_enable:

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


    ############ Bookkeeping functions

    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.position]))
        self.angle_hist = np.concatenate((self.angle_hist, [self.angle]))
        self.t.append(self.t[-1] + self.time_step)
        self.r_hist.append(self.reward())


    def __del__(self):
        del self.motor
        GPIO.cleanup()
