class ConstantRankSchedule:
    def __init__(self, max_rank: int, low_bound_lora: int=150, high_bound_lora: int=500, low_freq:int=5, high_freq:int=10):
        
        self.max_rank = max_rank
        self.low_bound_lora = low_bound_lora
        self.high_bound_lora = high_bound_lora
        self.low_freq = low_freq
        self.high_freq = high_freq

    def phase(self, step: int):
        if step < self.low_bound_lora:
            return True, self.max_rank
        elif step < self.high_bound_lora:
            if step % self.low_freq == 0:
                return True, self.max_rank
            else:
                return False, self.max_rank
        else:
            if step % self.high_freq == 0:
                return True, self.max_rank
            else:
                return False, self.max_rank
