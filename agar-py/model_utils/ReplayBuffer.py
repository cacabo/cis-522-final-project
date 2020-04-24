import random

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
    
    # save a state transition memory
    def push(self, memory):
        if len(self.buffer) < self.capacity:
            self.memory.append(None)
        self.buffer[self.idx] = memory
        self.idx = (self.idx + 1) % self.capacity

    # get a random sample of [batch_size] memories
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)