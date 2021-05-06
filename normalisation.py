from statistics import NormalDist
import tensorflow as tf

class normalise:
    def __init__(self, maxNumb=1000, ci=0.90):
        self.maxNumb = maxNumb
        self.data = [1, 1]
        self.dist = {'max': 1,
                     'min': -1}
        self.ci = ci
        self.range = {'max': 1,
                      'min': -1}

    def logNorm(self, step):

        if len(self.data) >= self.maxNumb:
            self.data = self.data[:self.maxNumb - 1]
            self.data.insert(0, float(step))

        else:
            self.data.insert(0, float(step))

        dist = NormalDist.from_samples(self.data)
        # z = NormalDist().inv_cdf((1 + self.ci) / 2.)
        # h = dist.stdev * z / ((len(self.data) - 1) ** .5)
        # self.dist = {'max': dist.mean + h,
        #              'min': dist.mean - h}
        #
        # rnge = self.dist['max'] - self.dist['min']

        return (step - dist.mean)/dist.stdev
