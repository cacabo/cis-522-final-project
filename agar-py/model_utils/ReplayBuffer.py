import random


class ReplayBuffer():
    def __init__(self, capacity, prefill_amt=1):
        self.capacity = capacity
        self.prefill_amt = prefill_amt
        self.buffer = []
        self.idx = 0

    def push(self, memory):
        """save a state transition memory"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = memory
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        """get a random sample of [batch_size] memories"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def prefill_capacity(self):
        return len(self.buffer) / self.prefill_amt