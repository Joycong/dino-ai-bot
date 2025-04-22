import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(experience)

    def sample(self, batch_size):
        import random
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Replay memory saved to {file_path}")

    def load(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Replay memory loaded from {file_path}")
        except FileNotFoundError:
            print(f"No replay memory file found at {file_path}, starting fresh.")
