class TrunkScheduler:
    def __init__(self, low_bound: int = 150, high_bound: int = 500, low_freq: int = 5, high_freq: int = 10):
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.low_freq = low_freq
        self.high_freq = high_freq

    def do_freeze(self, iteration)-> bool:
        #naive implementation for hopper, fully hardcoded: don't freeze before 150, then unfreeze every 5 iterations until 500, then unfreeze every 10 iterations
        if iteration < self.low_bound:
            return False
        elif iteration >= self.low_bound and iteration < self.high_bound and iteration % self.low_freq != 0:
            return True
        elif iteration >= self.high_bound and iteration % self.high_freq != 0:
            return True
        return False