import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # 옵티마이저 추가

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path, input_dim, output_dim):
    model = DQN(input_dim, output_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")
    else:
        print("No pre-trained model found. Initializing new model.")
    return model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print("Model saved.")