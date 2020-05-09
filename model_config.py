"""The class that defines certain constants of the model."""

import numpy as np

class Model:
    # nfft is 512 now since our rate has been downsampled (now the default)
    def __init__(
        self, 
        min=np.Inf,
        max=-np.Inf,
        X=0,
        y=0,
        mode="conv", 
        nfilt=26, 
        nfeat=13, 
        nfft=512, 
        rate=16000,
    ):
        self.min = min
        self.max = max
        self.data = (X, y)
        self.sample_size = 0.1
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate * self.sample_size)
