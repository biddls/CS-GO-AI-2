import tensorflow as tf
import NN
import time
from getdat import ctrls
import LDSV
from win32gui import GetWindowText, GetForegroundWindow
from util import Capture
from wanb import Wanb
from normalisation import normalise


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class agentBeginnerMouseOnlyTraining():

    def __init__(self, NNModel, FPS=False):
        self.model = NNModel
        self.did = None
        self.FPS = FPS

    def start(self):
        # self.restart()#restarts cs go game

        counter = 0
        start = time.time()
        logging = Wanb(time.asctime(), 0)
        norm = normalise()
        try:
            while 1:
                if GetWindowText(GetForegroundWindow()) == "Counter-Strike: Global Offensive":

                    observation = capture.getImg()  # grabs new screen data
                    # ignore fn puts part of it though a soft max
                    # self.did = ctrls.actionMouseOnly(self.model.predict(observation)[0])
                    self.did = ctrls.classification(self.model.predict(observation)[0])
                    while capture.getImg() is observation:
                        pass

                    # train model NN 1 step for each observation
                    self.model, error = NN.trainRL1Sample(self.model, observation, capture.getImg(), self.did, norm)
                    try:
                        logging.log(error)
                    except:
                        pass

                counter += 1
                if (time.time() - start) > 1 and self.FPS:
                    print("FPS: ", counter / (time.time() - start))
                    counter = 0
                    start = time.time()
                    pass

        # When i halt the training it runs through the close down process
        except KeyboardInterrupt:
            print('#############|Training Stopped|#############')
            logging.end()


if __name__ == '__main__':
    modelName = 'RLCS.h5'
    print("just give it a few seconds to warm up")

    model = LDSV.loadInit(modelName)
    # Start new Threads
    capture = Capture(1, 'image boi')
    capture.start()

    time.sleep(5)
    agent = agentBeginnerMouseOnlyTraining(model)
    agent.start()
    print('Go to: https://wandb.ai/thomasbiddlecombe/RL%20CS to watch the score')
    while 1:
        time.sleep(10)
        LDSV.saveWeight(agent.model, modelName)

# bot_kick; bot_knives_only 1; mp_roundtime 60; mp_maxrounds 100; sv_cheats 1; cl_pitchup 10; cl_pitchdown 10; mp_warmup_end; bot_add_t
