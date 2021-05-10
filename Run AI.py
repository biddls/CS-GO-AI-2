import tensorflow as tf
import time
from getdat import ctrls
import LDSV
from win32gui import GetWindowText, GetForegroundWindow
from util import Capture

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class testing:
    def __init__(self, NNModel):
        self.model = NNModel

    def start(self):
        try:
            while 1:
                if GetWindowText(GetForegroundWindow()) == "Counter-Strike: Global Offensive":
                    ctrls.classification(self.model.predict(capture.getImg())[0])
        except:
            pass


if __name__ == '__main__':
    modelName = 'RLCS.h5'
    print("just give it a few seconds to warm up")

    model = LDSV.loadInit(modelName)
    # Start new Threads
    capture = Capture(1, 'image boi')
    capture.start()

    time.sleep(5)
    agent = testing(model)
    agent.start()

# bot_kick; bot_knives_only 1; mp_roundtime 60; mp_maxrounds 100; sv_cheats 1; cl_pitchup 10; cl_pitchdown 10; mp_warmup_end; bot_add_t
