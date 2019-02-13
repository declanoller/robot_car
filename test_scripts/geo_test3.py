import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt


dist_meas_percent_tolerance = 0.2
wall_length = 1.25
'''xlims = np.array([-wall_length/2, wall_length/2])
ylims = np.array([-wall_length/2, wall_length/2])'''
xlims = np.array([-2, 2])
ylims = np.array([-2, 2])
position = np.array([0.5*(max(xlims) + min(xlims)), 0.5*(max(ylims) + min(ylims))])
bottom_corner = np.array([xlims[0], ylims[0]])



def touchingSameWall(a, a_theta, b, b_theta):
    #This tests if two vectors are touching the same wall, i.e, either of their coords are the same.
    #If any are, then it returns the coord and the index of it. Otherwise, returns None.
    #a and b are their magnitudes, the thetas are their angle w.r.t. the right-pointing horizontal.

    #x
    x1 = a*cos(a_theta)
    x2 = b*cos(b_theta)
    print('abs(x1 - x2): {:.3f}'.format(abs(x1 - x2)))
    same_coord_acc = abs((x1 - x2)/wall_length)
    print('same coord acc, x: {:.3f}'.format(same_coord_acc))
    #print('x1={:.4f}, x2={:.4f}, norm diff={}'.format(x1,x2, abs((x1 - x2)/(0.5*(x1 + x2)))))
    if same_coord_acc < dist_meas_percent_tolerance:
        return(x1, 0)

    #y
    y1 = a*sin(a_theta)
    y2 = b*sin(b_theta)
    print('abs(y1 - y2): {:.3f}'.format(abs(y1 - y2)))
    same_coord_acc = abs((y1 - y2)/wall_length)
    print('same coord acc, y: {:.3f}'.format(same_coord_acc))
    if same_coord_acc < dist_meas_percent_tolerance:
        return(y1, 1)

    return(None)


def touchingOppWall(a, a_theta, b, b_theta):
    #Returns index of two that are touching opp walls, None otherwise.
    #Also returns the coordinate we're sure about now, which is the negative of the
    #negative of the pair that makes up the span.

    #x
    x1 = a*cos(a_theta)
    x2 = b*cos(b_theta)
    span = abs(x1 - x2)
    print('x span: {:.3f}'.format(span))
    #print('span x={}'.format(span))
    #if abs((span - self.wall_length)/(0.5*(span + self.wall_length))) < self.dist_meas_percent_tolerance:
    span_accuracy = abs((span - wall_length)/wall_length)
    print('x span accuracy: {:.3f}'.format(span_accuracy))
    if span_accuracy < dist_meas_percent_tolerance:
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
    print('y span: {:.3f}'.format(span))
    span_accuracy = abs((span - wall_length)/wall_length) # Lower is better
    print('y span accuracy: {:.3f}'.format(span_accuracy))
    if span_accuracy < dist_meas_percent_tolerance:
        if y1 < 0:
            return(-y1, 1, span_accuracy)
        else:
            return(-y2, 1, span_accuracy)

    return(None)


def cornerOriginToCenterOrigin(pos):

    # This changes it so if your coords are so that the origin is in the
    # bottom left hand corner, now it's in the middle of the arena.

    center_origin_pos = pos + bottom_corner
    return(center_origin_pos)


def calculatePosition(d1, d2, d3, theta):

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

    print('\n\nsame wall:')
    print('1-2:')
    pair12_same = touchingSameWall(*pair12)
    print('2-3:')
    pair23_same = touchingSameWall(*pair23)
    print('1-3:')
    pair13_same = touchingSameWall(*pair13)

    print('\n\nopp walls:')
    print('1-2:')
    pair12_opp = touchingOppWall(*pair12)
    print('2-3:')
    pair23_opp = touchingOppWall(*pair23)
    print('1-3:')
    pair13_opp = touchingOppWall(*pair13)
    print('\n')
    sol = np.array([0.0, 0.0])
    same = (pair12_same is not None) or (pair23_same is not None) or (pair13_same is not None)
    opp = (pair12_opp is not None) or (pair23_opp is not None) or (pair13_opp is not None)



    if (same and not opp) or (opp and not same):
        if same and not opp:
            print('Touching same wall: pair12_same={}, pair23_same={}, pair13_same={}'.format(pair12_same, pair23_same, pair13_same))
            #print('two touching same wall')

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
                sol[coord] = wall_length - dist
            else:
                sol[coord] = -dist

        if opp and not same:
            print('Touching opp wall: pair12_opp={}, pair23_opp={}, pair13_opp={}'.format(pair12_opp, pair23_opp, pair13_opp))

            #This means that no two touch the same wall.
            #print('opp walls, not the same')

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

        print('dist={:.3f}, coord={}, other_coord={}, other_ray=({:.3f}, {:.3f})'.format(dist, coord, other_coord, other_ray[0], other_ray[1]))


        if other_dist>=0:
            sol[other_coord] = wall_length - other_dist
        else:
            sol[other_coord] = -other_dist

        return(cornerOriginToCenterOrigin(sol))

    if same and opp:
        #print('unsolvable case, touching same wall and spanning. Attempting best guess')
        print('Touching same AND opp walls.')

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
            sol[coord] = wall_length - dist
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

        sol[other_coord] = (-below_margin + (wall_length - above_margin + below_margin)/2.0)
        return(cornerOriginToCenterOrigin(sol))


    # This is if something is wrong and it can't figure out the position.
    print('Couldnt calc position based on meas., returning default val of (0,0) (in center origin coords).')
    return(sol)





#
# Hard case, touching same and opp walls probably.
d1=0.7
d2=0.22
d3=1.02
angle= -1.24

# pos. calcd in calcPosition=(-0.427, 0.082)
d1=0.939
d2=0.305
d3=0.263
angle=-0.853

d1=0.920
d2=0.306
d3=0.263
angle=-0.863

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
x,y = 0,0
pos = (x,y)
puck = plt.Circle(pos, 0.02, color='black')
ax.add_artist(puck)


x1, y1 = d1*cos(angle), d1*sin(angle)
x2, y2 = d2*cos(angle + pi/2), d2*sin(angle + pi/2)
x3, y3 = d3*cos(angle - pi/2), d3*sin(angle - pi/2)


tail = pos
tweak = 0.4
w = 0.002
ax.arrow(x, y, x1, y1, width=w, head_width=8*w, color='black')
ax.arrow(x, y, x2, y2, width=w, head_width=8*w, color='red')
ax.arrow(x, y, x3, y3, width=w, head_width=8*w, color='blue')


ax.axvline(x1, ylims[0], ylims[1], color='lightgray')
ax.axhline(y1, xlims[0], xlims[1], color='lightgray')
ax.axvline(x2, ylims[0], ylims[1], color='lightgray')
ax.axhline(y2, xlims[0], xlims[1], color='lightgray')
ax.axvline(x3, ylims[0], ylims[1], color='lightgray')
ax.axhline(y3, xlims[0], xlims[1], color='lightgray')


plt.show()






exit(0)

pos = calculatePosition(d1,d2,d3,ang)

print(pos)




fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
x, y = pos[0], pos[1]
puck = plt.Circle(pos, 0.02, color='tomato')
ax.add_artist(puck)

tail = pos
tweak = 0.4
w = 0.002
ax.arrow(x, y, d1*cos(ang), d1*sin(ang), width=w, head_width=8*w, color='black')
ax.arrow(x, y, d2*cos(ang + pi/2), d2*sin(ang + pi/2), width=w, head_width=8*w, color='red')
ax.arrow(x, y, d3*cos(ang - pi/2), d3*sin(ang - pi/2), width=w, head_width=8*w, color='blue')
plt.show()


exit(0)









#
