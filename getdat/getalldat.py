import multiprocessing
from getdat.gamestate_listener import ListenerWrapper


def GSIstart():
    # Message queue used for comms between processes
    queue = multiprocessing.Queue()
    listener = ListenerWrapper(queue)
    listener.start()

    # listener.shutdown()
    # listener.join()


if __name__ == "__main__":
    GSIstart()
