


### calculatePosition(), 2.24.2019, before edits


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

        self.df.writeToDebug('Same walls: pair12_same={}, pair23_same={}, pair13_same={}'.format(
        self.formatPosTuple(pair12_same),
        self.formatPosTuple(pair23_same),
        self.formatPosTuple(pair13_same)))
        self.df.writeToDebug('Opp walls: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(
        self.formatPosTuple(pair12_opp),
        self.formatPosTuple(pair23_opp),
        self.formatPosTuple(pair13_opp)))

        same_accs = [('12', pair12_same), ('13', pair13_same)]
        opp_accs = [('12', pair12_opp), ('23', pair23_opp), ('13', pair13_opp)]

        # This should sort for the lowest accuracy returned in the tuple.
        best_acc_tuple_same = sorted(same_accs, key=lambda x: x[1][2])[0]
        best_acc_tuple_opp = sorted(opp_accs, key=lambda x: x[1][2])[0]

        best_acc_same = best_acc_tuple_same[1][2]
        best_acc_opp = best_acc_tuple_opp[1][2]

        self.df.writeToDebug('Best accuracy, same wall: {}, {}'.format(
        best_acc_tuple_same[0],
        self.formatPosTuple(best_acc_tuple_same[1])))
        self.df.writeToDebug('Best accuracy, opp wall: {}, {}'.format(
        best_acc_tuple_opp[0],
        self.formatPosTuple(best_acc_tuple_opp[1])))

        #sol = np.array([0.0, 0.0])
        sol = -self.bottom_corner # Puts it at (.65, .65), lower left origin (so center of arena)


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


    def coordInBds(self, coord):
        # This calculates if an (x,y) np array is in the bounds (plus a little margin)
        # of the arena, with center origin coords. (So, it will consider (0.7, 0.1) to be
        # in bounds even though it's technically out.)
        if abs(coord[0]) > 1.1*self.wall_length/2:
            return(False)
        if abs(coord[1]) > 1.1*self.wall_length/2:
            return(False)

        return(True)







###############################################






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


        '''if same_coord_acc_x < self.dist_meas_percent_tolerance:
            return(x1, 0, same_coord_acc_x)


        if same_coord_acc_y < self.dist_meas_percent_tolerance:
            return(y1, 1, same_coord_acc_y)

        return(None)'''


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
        span_accuracy_x = abs((span - self.wall_length)/self.wall_length)

        #y
        y1 = a*sin(a_theta)
        y2 = b*sin(b_theta)
        span = abs(y1 - y2)
        span_accuracy_y = abs((span - self.wall_length)/self.wall_length) # Lower is better

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

        '''if span_accuracy_x < self.dist_meas_percent_tolerance:
            # This current way of doing it anchors it to the LEFT/DOWN, so it will always have that bias.
            # Instead let's try taking the average.
            if x1 < 0:
                return(-x1, 0, span_accuracy_x)
            else:
                return(-x2, 0, span_accuracy_x)


        if span_accuracy_y < self.dist_meas_percent_tolerance:
            if y1 < 0:
                return(-y1, 1, span_accuracy_y)
            else:
                return(-y2, 1, span_accuracy_y)

        return(None)'''


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

        best_acc_same =


        sol = np.array([0.0, 0.0])
        same = (pair12_same is not None) or (pair23_same is not None) or (pair13_same is not None)
        opp = (pair12_opp is not None) or (pair23_opp is not None) or (pair13_opp is not None)

        if (same and not opp) or (opp and not same):
            if same and not opp:
                #print('two touching same wall')
                self.df.writeToDebug('Touching same wall: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))

                if pair12_same is not None:
                    dist, coord, acc = pair12_same
                    other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

                if pair23_same is not None:
                    dist, coord, acc = pair23_same
                    other_ray = [d1*cos(theta), d1*sin(theta)]

                if pair13_same is not None:
                    dist, coord, acc = pair13_same
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
                dist, coord, acc = pair12_same
                other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

            if pair23_same is not None:
                dist, coord, acc = pair23_same
                other_ray = [d1*cos(theta), d1*sin(theta)]

            if pair13_same is not None:
                dist, coord, acc = pair13_same
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
