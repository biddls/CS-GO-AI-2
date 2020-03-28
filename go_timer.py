import multiprocessing
from gamestate_listener import ListenerWrapper
from screen_grab import grab_screen
import numpy as np
import time

def main():
    # Message queue used for comms between processes
    queue = multiprocessing.Queue()
    listener = ListenerWrapper(queue)
    listener.start()

    path = 'C:\\Users\\thoma\\Downloads\\GoTimer-master\\data'
    count = 0
    while True:
        count += 1
        np.save(path + '\\' + str(count) +'.np', np.array([round(time.time()*100000000), grab_screen()]))
    #listener.shutdown()
    #listener.join()

if __name__ == "__main__":
    main()