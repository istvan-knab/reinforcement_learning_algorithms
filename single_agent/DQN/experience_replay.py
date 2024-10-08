from collections import deque, namedtuple
import random
import torch

class ReplayMemory:
    def __init__(self, buffer_size:int, batch_size:int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=self.buffer_size)
        random.seed(0)

    def add_element(self, *args):
        Transition = namedtuple('Transition',('state', 'action',
                                              'next_state', 'reward', 'done'))
        self.memory.append(Transition(*args))

    def sample(self):
        self.batch_size = random.randint(256, 512)
        batch = zip(*(random.sample(self.memory, self.batch_size)))
        return [torch.cat(items) for items in batch]

    def __len__(self):
        return len(self.memory)

