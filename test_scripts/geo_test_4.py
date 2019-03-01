import numpy as np
from math import sin, cos, tan, pi


wall_length = 1.25
xlims = np.array([-wall_length/2, wall_length/2])
ylims = np.array([-wall_length/2, wall_length/2])
position = np.array([0.5*(max(xlims) + min(xlims)), 0.5*(max(ylims) + min(ylims))])
last_position = position
bottom_corner = np.array([xlims[0], ylims[0]])

dist_meas_percent_tolerance = 0.05

def touchingSameWall(a, a_theta, b, b_theta):
    #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
    #If any are, then it returns the coord and the index of it. Otherwise, returns None.
    #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.

    #x
    x1 = a*cos(a_theta)
    x2 = b*cos(b_theta)
    #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
    #same_coord_acc_x = abs((x1 - x2)/wall_length)
    same_coord_acc_x = abs(x1 - x2)/(0.5*(abs(x1) + abs(x2)))

    #y
    y1 = a*sin(a_theta)
    y2 = b*sin(b_theta)
    #same_coord_acc_y = abs((y1 - y2)/wall_length)
    same_coord_acc_y = abs(y1 - y2)/(0.5*(abs(y1) + abs(y2)))

    if same_coord_acc_x < same_coord_acc_y:
        return(x1, 0, same_coord_acc_x)
    else:
        return(y1, 1, same_coord_acc_y)


def touchingOppWall(a, a_theta, b, b_theta):
    #Returns index of two that are touching opp walls, None otherwise.
    #Also returns the coordinate we're sure about now, which is the negative of the
    #negative of the pair that makes up the span.

    #x
    x1 = a*cos(a_theta)
    x2 = b*cos(b_theta)
    span_x = abs(x1 - x2)
    #span_x = abs(x1) + abs(x2)
    #print('span x={}'.format(span))
    #if abs((span - wall_length)/(0.5*(span + wall_length))) < dist_meas_percent_tolerance:
    span_accuracy_x = abs((span_x - wall_length)/wall_length)

    #y
    y1 = a*sin(a_theta)
    y2 = b*sin(b_theta)
    span_y = abs(y1 - y2)
    #span_y = abs(y1) + abs(y2)
    span_accuracy_y = abs((span_y - wall_length)/wall_length) # Lower is better

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


def posAngleToCollideVec(pos, ang):
    #
    # This takes a proposed position and angle, and returns the vector
    # that would collide with the wall it would hit at that angle.
    #
    # This assumes *lower left* origin coords.
    #
    # The angle bds here are the 4 angle bounds that determine
    # for a given x,y which wall the ray in that direction would collide with.
    # The order is: top wall, left wall, bottom wall, right wall

    x = pos[0]
    y = pos[1]

    angle_bds = [
    np.arctan2(wall_length-y, wall_length-x),
    np.arctan2(wall_length-y, -x),
    np.arctan2(-y, -x),
    np.arctan2(-y, wall_length-x),
    ]

    ray_x, ray_y = 0, 0

    if (ang >= angle_bds[0]) and (ang < angle_bds[1]):
        ray_y = wall_length - y
        ray_x = ray_y/tan(ang)

    elif (ang >= angle_bds[1]) or (ang < angle_bds[2]):
        ray_x = -x
        ray_y = ray_x*tan(ang)

    elif (ang >= angle_bds[2]) and (ang < angle_bds[3]):
        ray_y = -y
        ray_x = ray_y/tan(ang)

    elif (ang >= angle_bds[3]) and (ang < angle_bds[0]):
        ray_x = wall_length - x
        ray_y = ray_x*tan(ang)

    return(np.array([ray_x, ray_y]))


def formatPosTuple(tup):

    if tup is None:
        return(None)
    else:
        return('({:.3f}, {}, {:.3f})'.format(*tup))


def cornerOriginToCenterOrigin(pos):

    # This changes it so if your coords are so that the origin is in the
    # bottom left hand corner, now it's in the middle of the arena.

    center_origin_pos = pos + bottom_corner
    return(center_origin_pos)


def calculatePosition(d1, d2, d3, theta):

    print('************************* In function: {}()'.format('calculatePosition'))
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

    pair12_same = touchingSameWall(*pair12)
     # 2 and 3 should never be able to hit the same wall... and it makes trouble when they hit opposite walls at the same height!
    #pair23_same = touchingSameWall(*pair23)
    pair23_same = None
    pair13_same = touchingSameWall(*pair13)


    pair12_opp = touchingOppWall(*pair12)
    pair23_opp = touchingOppWall(*pair23)
    pair13_opp = touchingOppWall(*pair13)

    print('Same walls: pair12_same={}, pair23_same={}, pair13_same={}'.format(
    formatPosTuple(pair12_same),
    formatPosTuple(pair23_same),
    formatPosTuple(pair13_same)))
    print('Opp walls: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(
    formatPosTuple(pair12_opp),
    formatPosTuple(pair23_opp),
    formatPosTuple(pair13_opp)))

    errors = [
    ('12', pair12_same, 'same'),
    ('13', pair13_same, 'same'),
    ('12', pair12_opp, 'opp'),
    ('23', pair23_opp, 'opp'),
    ('13', pair13_opp, 'opp')
    ]

    print('\nerrors:')
    print(errors)


    #sol = np.array([0.0, 0.0])

    pos_err_list = []

    print('\n\n')
    for err in errors:

        sol = -bottom_corner # Puts it at (.65, .65), lower left origin (so center of arena)
        pair_label = err[0]
        dist, coord, acc = err[1]
        match_label = err[2]
        print('\nerr: ', err)

        if match_label == 'same':
            #print('Touching same wall: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))


            if pair_label == '12':
                other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

            if pair_label == '23':
                other_ray = [d1*cos(theta), d1*sin(theta)]

            if pair_label == '13':
                other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

            #Sets the coordinate we've figured out.
            if dist>=0:
                sol[coord] = wall_length - dist
            else:
                sol[coord] = -dist

        if match_label == 'opp':
            #This means that no two touch the same wall.
            #print('opp walls, not the same')
            #print('Touching opp wall: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(pair12_opp, pair23_opp, pair13_opp))

            if pair_label == '12':
                other_ray = [d3*cos(theta - pi/2), d3*sin(theta - pi/2)]

            if pair_label == '23':
                other_ray = [d1*cos(theta), d1*sin(theta)]

            if pair_label == '13':
                other_ray = [d2*cos(theta + pi/2), d2*sin(theta + pi/2)]

            #The dist should already be positive.
            sol[coord] = dist

        #This is the other coord we don't have yet, which works for either case.
        other_coord = abs(1 - coord)
        other_dist = other_ray[other_coord]

        print('dist={:.3f}, coord={}, other_coord={}, other_ray=({:.3f}, {:.3f})'.format(dist, coord, other_coord, other_ray[0], other_ray[1]))

        if other_dist>=0:
            sol[other_coord] = wall_length - other_dist
        else:
            sol[other_coord] = -other_dist

        #pos = cornerOriginToCenterOrigin(sol)
        pos = sol
        print('pos. calcd in calcPosition=({:.3f}, {:.3f})'.format(pos[0], pos[1]))

        d1_collide_vec = posAngleToCollideVec(pos, theta)
        d2_collide_vec = posAngleToCollideVec(pos, theta + pi/2)
        d3_collide_vec = posAngleToCollideVec(pos, theta - pi/2)

        d1_vec = d1*np.array([cos(theta + 0), sin(theta + 0)])
        d2_vec = d2*np.array([cos(theta + pi/2), sin(theta + pi/2)])
        d3_vec = d3*np.array([cos(theta - pi/2), sin(theta - pi/2)])

        print('d1 vec and collide:', d1_vec, d1_collide_vec)
        print('d2 vec and collide:', d2_vec, d2_collide_vec)
        print('d3 vec and collide:', d3_vec, d3_collide_vec)

        tot_err = np.linalg.norm(d1_vec - d1_collide_vec) + np.linalg.norm(d2_vec - d2_collide_vec) + np.linalg.norm(d3_vec - d3_collide_vec)

        pos_err_list.append([pos, tot_err, pair_label, match_label])
        #return(pos)

    print('\n\n\n')
    [print(err) for err in pos_err_list]
    # This is if something is wrong and it can't figure out the position.
    print('Couldnt calc position based on meas., returning default val of (0,0) (in center origin coords).')
    return(sol)








p = calculatePosition(.18, .995, .33, .83)
print('\n\npos found: ', p)















#
