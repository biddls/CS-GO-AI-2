import threading
import time
from getdat import getalldat
from getdat import ctrls

###plan###
# get image in
# load in GSI

"""path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\images'
    count = 0

    while True:
        count += 1
        np.save(path + '\\' + str(count) + '.npy', np.array([round(time.time() * 100000000), grab_screen()]))"""


class myThread(threading.Thread):
    def __init__(self, threadID, name, ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt'
        self.file = None

    def run(self):
        print("Starting " + self.name)
        self.file = getupdate(self.path)


def getupdate(path):
    try:
        file = open(path)
        return file
    except:
        print('doesnt exist')


if __name__ == '__main__':
    getalldat.GSIstart()
    thread1 = myThread(1, "Thread-1")

    # Start new Threads
    thread1.start()
