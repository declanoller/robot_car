import curses


def loop(stdscr):
    while True:
        c = stdscr.getch()
        if c == ord('p'):
            print('you pressed p!')
        elif c == ord('q'):
            print('you pressed q! exiting')
            break  # Exit the while loop
        elif c == curses.KEY_HOME:
            x = y = 0

curses.wrapper(loop)
