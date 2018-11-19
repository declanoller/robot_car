
import curses
import time
import numpy as np


def DCloop(stdscr):
    #https://docs.python.org/3/howto/curses.html
    #https://docs.python.org/3/library/curses.html#curses.window.clrtobot
    delay_time = 0.0
    print(curses.LINES) #Lines go from left to right, so this is the max y coord.
    print(curses.COLS) #Max X coord.
    #For each angle, the list is forward and back.
    angle_dir_dict = {
                        0: np.array([1,0]),
                        90: np.array([0,-1]),
                        180: np.array([-1,0]),
                        270: np.array([0,1])
                        }


    box_side_y = 20
    box_side_x = 2*box_side_y

    box_coord_y = 5
    box_coord_x = curses.COLS - (box_side_x + 5)

    pos_y = box_coord_y + int(box_side_y/2)
    pos_x = box_coord_x + int(box_side_x/2)
    pos = np.array([pos_x, pos_y])
    angle = 0

    box_info = (box_coord_y, box_coord_x, box_side_y, box_side_x)
    drawAllStandard(stdscr, pos, angle, box_info)
    stdscr.addstr(pos_y, pos_x, '→')
    stdscr.move(0,0)
    while True:
        c = stdscr.getch()
        if c == curses.KEY_LEFT:
            if pos_x > (box_coord_x + 1):
                drawAllStandard(stdscr, pos, angle, box_info)
                '''pos_x += -1
                drawCartesianPosition(stdscr, pos)'''
                angle += 90
                angle = angle%360
                drawAnglePosition(stdscr, pos, angle)

        if c == curses.KEY_RIGHT:
            if pos_x < (box_coord_x + box_side_x - 1):
                drawAllStandard(stdscr, pos, angle, box_info)
                '''pos_x += 1
                drawCartesianPosition(stdscr, pos)'''
                angle += -90
                angle = angle%360
                drawAnglePosition(stdscr, pos, angle)

        if c == curses.KEY_UP:
            if pos_y > (box_coord_y + 1):
                pos += angle_dir_dict[angle]
                drawAllStandard(stdscr, pos, angle, box_info)
                '''pos_y += -1
                drawCartesianPosition(stdscr, pos)'''
                drawAnglePosition(stdscr, pos, angle)

        if c == curses.KEY_DOWN:
            if pos_y < (box_coord_y + box_side_y - 1):
                pos += -angle_dir_dict[angle]
                drawAllStandard(stdscr, pos, angle, box_info)
                '''pos_y += 1
                drawCartesianPosition(stdscr, pos)'''
                drawAnglePosition(stdscr, pos, angle)

        #elif c == ord('q'):
        elif c == ord('q') or c == 27:
            print('you pressed q! exiting')
            break  # Exit the while loop


def drawCartesianPosition(stdscr, pos):
    (pos_x, pos_y) = (pos[0], pos[1])
    stdscr.addstr(pos_y, pos_x, '0')
    stdscr.move(0,0)


def drawAnglePosition(stdscr, pos, angle):
    (pos_x, pos_y) = (pos[0], pos[1])
    if (angle >= 315) or (angle < 45):
        symbol = '→'
    if (angle >= 45) and (angle < 135):
        symbol = '↑'
    if (angle >= 135) and (angle < 225):
        symbol = '←'
    if (angle >= 225) and (angle < 315):
        symbol = '↓'
    stdscr.addstr(pos_y, pos_x, symbol)
    #stdscr.refresh()
    stdscr.move(0,0)



def drawAllStandard(stdscr, pos, angle, box_info):

    stdscr.erase() #Use this one supposedly https://stackoverflow.com/questions/9653688/how-to-refresh-curses-window-correctly
    redrawBox(stdscr, box_info)
    drawPosInfo(stdscr, pos, angle, box_info)
    stdscr.addstr(curses.LINES - 1, 0,  'Press q or Esc to quit')
    stdscr.move(0,0)
    stdscr.refresh()


def redrawBox(stdscr, box_info):
    side_symbol = '@'
    (box_coord_y, box_coord_x, box_side_y, box_side_x) = box_info
    stdscr.addstr(box_coord_y, box_coord_x, side_symbol*box_side_x)
    stdscr.addstr(box_coord_y + box_side_y, box_coord_x, side_symbol*box_side_x)
    for i in range(box_side_y+1):
        stdscr.addstr(box_coord_y + i, box_coord_x, side_symbol)
        stdscr.addstr(box_coord_y + i, box_coord_x + box_side_x, side_symbol)


def drawPosInfo(stdscr, pos, angle, box_info):
    (pos_x, pos_y) = (pos[0], pos[1])
    (box_coord_y, box_coord_x, box_side_y, box_side_x) = box_info

    info_str = 'Meas. position: (x = {:.2f}, y = {:.2f})\t Meas. angle: {}°'.format(pos_x, pos_y, angle)
    stdscr.addstr(0, 30,  info_str)



def directControl():
    print('entering curses loop')
    curses.wrapper(DCloop)
    print('exited curses loop.')



directControl()
