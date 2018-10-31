import time
import RPi.GPIO as GPIO
from Motor import Motor
from Sonar import Sonar
from Compass import Compass

from random import randint, random, sample, choice
import numpy as np
from math import sin, cos, tan, pi


class Robot:

    def __init__(self, motor_enable=True, sonar_enable=True, compass_enable=True):


        #GPIO Mode (BOARD / BCM)
        GPIO.setmode(GPIO.BOARD)


        self.motor_enable = motor_enable
        self.sonar_enable = sonar_enable
        self.compass_enable = compass_enable

        if self.motor_enable:
            self.motor = Motor(left_forward_pin=33, left_reverse_pin=31, right_forward_pin=37, right_reverse_pin=35)
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

        self.policy_NN = None
        self.N_state_terms = 6
        self.params['N_hidden_layer_nodes'] = 50
        self.params['NL_fn'] = 'tanh'
        self.params['epsilon'] = 0.05
        self.params['epsilon_decay'] = 0.99
        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.N_actions = 4
        self.arena_side = 1.0

        #These will be the positions in meters, where (0,0) is the center of the arena.
        self.target_positions = [(-0.5, -0.5), (0.5, 0.5)]
        self.N_targets = len(self.target_positions)
        self.current_target = 0





    def updateEpsilon(self):
        self.params['epsilon'] *= self.params['epsilon_decay']


    def getStateVec(self):

        assert self.sonar_enable, 'No sonar! exiting.'
        assert self.compass_enable, 'No compass! exiting.'

        d1 = self.sonar_forward.distance()
        d2 = self.sonar_left.distance()
        d3 = self.sonar_right.distance()
        compass_reading = self.compass.getReading()[2] #Is it 2? figure out.










    def epsGreedyAction(self, state):

        assert self.policy_NN is not None, 'No NN to get action from! crashing.'

        if random()>self.params['epsilon']:
            return(self.greedyAction(state_vec))
        else:
            return(self.getRandomAction())


    def getRandomAction(self):
        return(randint(0,self.N_actions-1))


    def forwardPassQ(self, state_vec):
        return(self.policy_NN(state_vec))


    def singleStateForwardPassQ(self, state_vec):
        qsa = torch.squeeze(self.forwardPassQ(torch.unsqueeze(torch.Tensor(state_vec), dim=0)))
        return(qsa)

    def greedyAction(self,state_vec):
        qsa = self.singleStateForwardPassQ(state_vec)
        return(torch.argmax(qsa))



    def takeAction(self, action):

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




    def getReward(self):

        triggered_IR_sensors = self.pollTargetServer()

        if triggered_IR_sensors[self.current_target]==1:
            other_sensors = [i for i in range(self.N_targets) if i!=self.current_target]
            self.current_target = random.choice(other_sensors)
            return(1.0)
        else:
            return(-0.01)



    def pollTargetServer(self):
        #stuff...
        return([0]*self.N_targets)


    def directControlFromModel(self, model_fname, N_steps=10**3):

        import torch
        from DQN import DQN

        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)

        D_in, H, D_out = self.N_state_terms, self.params['N_hidden_layer_nodes'], self.N_actions

        NL_fn_dict = {'relu':F.relu, 'tanh':torch.tanh, 'sigmoid':F.sigmoid}
        NL_fn = NL_fn_dict[self.params['NL_fn']]

        self.policy_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)

        self.policy_NN.load_state_dict(torch.load(model_fname))

        self.params['epsilon'] = 0

        for i in range(N_steps):

            if i%int(N_steps/10) == 0:
                print('iteration ',i)

            self.updateEpsilon()

            s = self.getStateVec()
            a = self.epsGreedyAction(s)
            r, s_next = self.takeAction(a)

            self.R_tot += r.item()
            self.R_tot_hist.append(self.R_tot/(i+1))

            if r.item() > 0:
                self.resetTarget()


        print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/N_steps))
        return(self.R_tot/N_steps)





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


    def getPosition(self, d1, d2, d3, theta):

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





    def testAllDevices(self):

        if self.motor_enable:
            print('testing motor!')
            self.motor.wheelTest(test_time=5)
        if self.sonar_enable:
            print('testing sonar!')
            self.sonar_forward.distanceTestLoop(test_time=3)
        if self.compass_enable:
            print('testing compass!')
            #Compass uses I2C pins, which are 3 and 5 for the RPi 3.
            self.compass.readCompassLoop(test_time=4)






    def __del__(self):
        GPIO.cleanup()
