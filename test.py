from getdat import ctrls
from time import sleep

x = 0.2
y = 0.00
# ranges are: 0.01, 0.05, 0.2

sleep(3)
while 1:
    sleep(1)
    ctrls.moveMouse(1920 * x, 1080 * y)

# X&Y, 3 for each and do nothing = 8 outputs
