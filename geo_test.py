from math import cos, sin, tan, pi
import matplotlib.pyplot as plt



def touchingSameWall(a, a_theta, b, b_theta):
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



def touchingOppWall(a, a_theta, b, b_theta):
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







def testGeo(x, y, d1, d2, d3, theta):

    if (theta >= 0) and (theta < pi/2.0):
        quadrant = 1
    if (theta >= pi/2.0) and (theta < pi):
        quadrant = 2
    if (theta >= pi) and (theta < 3*pi/2.0):
        quadrant = 3
    if (theta >= 3*pi/2.0) and (theta < 2*pi):
        quadrant = 4


    print('quadrant:', quadrant)

    a = 1.0

    x1 = d1*cos(theta)
    y1 = d1*sin(theta)


    print('x1={:.3f}, y1={:.3f}'.format(x1,y1))

    #This is the angle of the triangle formed by connecting d1 to its corner.
    theta_prime = theta % (pi/2.0)

    print('theta (units of pi):', theta/pi)
    print('theta_prime (units of pi):', theta_prime/pi)

    #This is the hypotenuse of the triangle formed by d1, a ray in the direction of d2, and the
    #straight line connecting them (to see which wall d2 hits in reality)
    H = d1/sin(pi - theta)
    print('H:', H)

    #This means that d2 will be limited by the wall closer to d1.

    if H<a:
        d2limit = d1/tan(pi - theta)
    else:
        d2limit = (a - abs(y1))/cos(pi - theta)

    print('\n\nd2limit:', d2limit)

    x3 = d3*cos(theta - pi/2)
    y3 = d3*sin(theta - pi/2)


    #d2limit is the size such that if it's any BIGGER than this, it will be on the other solution side.
    if d2<1.1*d2limit:
        print('d2 is on left branch')
        x2 = d2*cos(theta + pi/2)
        y2 = d2*sin(theta + pi/2)
        print('x2={:.3f}, y2={:.3f}'.format(x2,y2))

        if abs(x1 - x2)/abs(x1) < 0.05:
            #In this case, d1 and d2 are hitting the same wall.
            print('d1 and d2 hitting the same side!')
            sol_x = abs(x1)

            sol_y = 1 - abs(y3)
        else:
            print('d1 and d2 hitting different sides!')
            #I think in this case, it's actually uniquely specified with only d1 and d2.
            #In this case, d1 and d2 are hitting adjacent, different walls.
            sol_x = abs(x1)
            sol_y = abs(y2)

    else:
        print('d2 is on right branch')

        x3 = d3*cos(theta + pi/2)
        y3 = d3*sin(theta + pi/2)
        print('x3={:.3f}, y3={:.3f}'.format(x3,y3))

        if abs(y1 - y3)/abs(y1) < 0.05:
            #In this case, d1 and d2 are hitting the same wall.
            print('d1 and d2 hitting the same side!')
            sol_y = abs(y1)

            sol_y = 1 - abs(y3)
        else:
            print('d1 and d2 hitting different sides!')
            #I think in this case, it's actually uniquely specified with only d1 and d2.
            #In this case, d1 and d2 are hitting adjacent, different walls.
            sol_x = abs(x1)
            sol_y = abs(y2)






    print('solution: x={}, y={}'.format(sol_x, sol_y))


    fig, ax = plt.subplots(1,1, figsize=(6,6))

    ax.set_xlim((0,1))
    ax.set_ylim((0,1))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    xy = (x,y)
    puck = plt.Circle(xy, 0.02, color='tomato')
    ax.add_artist(puck)

    tail = xy
    tweak = 0.4
    w = 0.002
    ax.arrow(x, y, x1*.95, y1*.95, width=w, head_width=8*w, color='black')
    ax.arrow(x, y, x2*.95, y2*.95, width=w, head_width=8*w, color='black')
    ax.arrow(x, y, x3*.95, y3*.95, width=w, head_width=8*w, color='black')
    plt.show()


def testGeo2(d1, d2, d3, theta):

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





d1 = 0.3
theta = 5*pi/6

#span only
'''y = .2
x = d1*abs(cos(theta))
d2 = abs(y)/abs(cos(theta))
d3 = (1 - y)/sin(theta - pi/2)'''

#same
'''x = .4
y = 1 - d1*abs(sin(theta))
d2 = abs(x)/abs(cos(theta + pi/2))
d3 = (1 - y)/sin(theta - pi/2)'''

#same AND span
x = .7
y = 1 - d1*abs(sin(theta))
d2 = abs(y)/abs(cos(theta))
d3 = (1 - y)/sin(theta - pi/2)

print('x={:.3f}, y={:.3f}, d1={:.3f}, d2={:.3f}, d3={:.3f}, theta={:.3f}'.format(x, y, d1, d2, d3, theta))


coords = testGeo2(d1, d2, d3, theta)
print('coords determined are: ({:.3f}, {:.3f})'.format(coords[0], coords[1]))


x1 = d1*cos(theta)
y1 = d1*sin(theta)
x2 = d2*cos(theta + pi/2)
y2 = d2*sin(theta + pi/2)
x3 = d3*cos(theta - pi/2)
y3 = d3*sin(theta - pi/2)

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.set_xlim((0,1))
ax.set_ylim((0,1))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

xy = (x,y)
puck = plt.Circle(xy, 0.02, color='tomato')
ax.add_artist(puck)

tail = xy
tweak = 0.4
w = 0.002
ax.arrow(x, y, x1*.98, y1*.98, width=w, head_width=8*w, color='black')
ax.arrow(x, y, x2*.98, y2*.98, width=w, head_width=8*w, color='black')
ax.arrow(x, y, x3*.98, y3*.98, width=w, head_width=8*w, color='black')
plt.show()

#testGeo(x, y, d1, d2, d3, theta)












'''#Figures out which is the wall that the two hit. This is needed to figure out if the third ray
#actually uniquely specifies everything or not.
#walls 1,2,3,4 are walls R,U,L,D.
if coord==0:
    if dist>=0:
        wall_same = 1
    else:
        wall_same = 3
else:
    if dist>=0:
        wall_same = 2
    else:
        wall_same = 4'''

'''

    print('pair12 same: ', pair12_same)
    print('pair23 same: ', pair23_same)
    print('pair13 same: ', pair13_same)

    print('pair12 opp: ', pair12_opp)
    print('pair23 opp: ', pair23_opp)
    print('pair13 opp: ', pair13_opp)'''


#
