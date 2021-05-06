import wandb as wandb


# this is just a few functions to manage the 'Weights and Bias' service in the back round
class Wanb:
    def __init__(self, name, startingStep):
        self.step = startingStep
        wandb.init(project='RL CS', config={
            'name': name})
        wandb.run.name = name

    def log(self, Error):
        wandb.log({'Error': Error}, self.step)
        self.step += 1

    def end(self):
        wandb.finish()

# test script
if __name__ == '__main__':
    import time
    import random
    env = Wanb(time.time())

    for x in range(1000):
        time.sleep(1)
        env.log(random.random())

    env.end()
