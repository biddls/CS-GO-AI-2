from statistics import NormalDist

class normalise:
    def __init__(self, maxNumb=1000):
        self.maxNumb = maxNumb
        self.data = [1, 1]
        self.dist = {'max': 1,
                     'min': -1}
        self.range = {'max': 1,
                      'min': -1}

    def logNorm(self, step):

        if len(self.data) >= self.maxNumb:
            self.data = self.data[:self.maxNumb - 1]
            self.data.insert(0, float(step))

        else:
            self.data.insert(0, float(step))

        dist = NormalDist.from_samples(self.data)
        return (step - dist.mean)/dist.stdev
