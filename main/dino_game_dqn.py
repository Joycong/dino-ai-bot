import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn_model import DQN
from dino_replay_memory import ReplayMemory
from dino_game_env import DinoGameEnv
import os
import csv

# Constants
MODEL_DIR = "../models"
MEMORY_DIR = "../memory"
LOG_PATH = "../logs/training_log.csv"

# 디렉토리 생성 (없을 경우 자동 생성)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

GAMMA = 0.99
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
BATCH_SIZE = 32
TOTAL_EPISODES = 1000
INITIAL_EPSILON = 0.9

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "Epsilon", "MaxObstacles"])

replay_memory = ReplayMemory(10000)


class DinoDQNAgent:
    def __init__(self, state_size, action_size, memory):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.good_memory = []
        self.good_memory_max = 1000
        self.epsilon = INITIAL_EPSILON
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        good_ratio = 0.7
        good_batch_size = int(self.batch_size * good_ratio)
        normal_batch_size = self.batch_size - good_batch_size

        good_samples = self.good_memory[-good_batch_size:]
        if len(good_samples) < good_batch_size:
            needed = good_batch_size - len(good_samples)
            normal_batch_size += needed
            good_samples = self.good_memory  # 가능한 만큼만 사용

        normal_samples = self.memory.sample(normal_batch_size)
        batch = good_samples + normal_samples

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, model_path, memory_path):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "epsilon": self.epsilon},
            model_path,
        )
        self.memory.save(memory_path)

    def load(self, model_path, memory_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["model_state_dict"])
            self.epsilon = checkpoint.get("epsilon", INITIAL_EPSILON)
        if os.path.exists(memory_path):
            self.memory.load(memory_path)


def get_latest_files():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("dqn_model_ep")]
    memory_files = [f for f in os.listdir(MEMORY_DIR) if f.startswith("replay_ep")]

    def extract_ep(filename, prefix):
        try:
            return int(filename.replace(prefix, "").split(".")[0])
        except:
            return -1

    model_eps = [extract_ep(f, "dqn_model_ep") for f in model_files]
    memory_eps = [extract_ep(f, "replay_ep") for f in memory_files]
    valid_eps = set(model_eps).intersection(set(memory_eps))

    if not valid_eps:
        return None, None, 1

    latest = max(valid_eps)
    return (
        os.path.join(MODEL_DIR, f"dqn_model_ep{latest}.pth"),
        os.path.join(MEMORY_DIR, f"replay_ep{latest}.pkl"),
        latest + 1,
    )


LATEST_MODEL_PATH, LATEST_MEMORY_PATH, START_EPISODE = get_latest_files()
env = DinoGameEnv()
agent = DinoDQNAgent(state_size=80 * 80, action_size=3, memory=replay_memory)

if LATEST_MODEL_PATH and LATEST_MEMORY_PATH:
    agent.load(LATEST_MODEL_PATH, LATEST_MEMORY_PATH)

max_obstacles = 0

for episode in range(START_EPISODE, START_EPISODE + TOTAL_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    episode_obstacles = 0
    step_count = 0
    time.sleep(0.1)

    print(f"--- 에피소드 {episode} 시작 ---")

    while not done:
        action = agent.act(state)
        env.last_action = action  # 마지막 행동을 env에 기록
        next_state, reward, done = env.step(action)


        agent.memory.add(state, action, reward, next_state, done)

        if reward > 0:  # 보상이 양수일 경우
            if len(agent.good_memory) < agent.good_memory_max:
                print("🟡 좋은 메모리 추가됨 (성공 사례)")
            else:
                print("🟠 좋은 메모리 갱신됨 (가장 오래된 항목 제거 후 추가)")
            agent.good_memory.append((state, action, reward, next_state, done))

        if len(agent.good_memory) > agent.good_memory_max:
            agent.good_memory = agent.good_memory[-agent.good_memory_max :]

        if reward == 10:
            episode_obstacles += 1

        agent.replay()
        state = next_state
        total_reward += reward
        step_count += 1

        print(
            f"[스텝 {step_count}] 행동: {action}, 보상: {reward:+}, 종료: {done}, ε: {agent.epsilon:.4f}"
        )

    print(f"\n💀 공룡 사망! 에피소드 {episode} 종료")
    print(f"✅ 총 보상: {total_reward:+} | 장애물 넘은 수: {episode_obstacles}")
    print(f"📉 ε: {agent.epsilon:.4f} | 🔼 기준: {max_obstacles}")
    print("-" * 50)

    if episode_obstacles > 0 and episode_obstacles >= max_obstacles:
        diff = episode_obstacles - max_obstacles
        decay_multiplier = 1.0 + diff * 0.2 if diff > 0 else 0.05
        new_eps = max(MIN_EPSILON, agent.epsilon * (EPSILON_DECAY**decay_multiplier))

        if episode_obstacles > max_obstacles:
            print(
                f"🌟 최대 갱신: {episode_obstacles} → 탐험률 대폭 감소 ({agent.epsilon:.4f} → {new_eps:.4f})"
            )
            max_obstacles = episode_obstacles
        else:
            print(
                f"🟢 동일한 성과: {episode_obstacles} → 탐험률 소폭 감소 ({agent.epsilon:.4f} → {new_eps:.4f})"
            )

        agent.epsilon = new_eps
    else:
        old_eps = agent.epsilon
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * 0.999)
        if agent.epsilon != old_eps:
            print(f"🕸️ 탐험률 미미하게 감소 ({old_eps:.4f} → {agent.epsilon:.4f})")

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, round(agent.epsilon, 4), max_obstacles])

    if episode % 10 == 0:
        agent.update_target_model()
        agent.save(
            os.path.join(MODEL_DIR, f"dqn_model_ep{episode}.pth"),
            os.path.join(MEMORY_DIR, f"replay_ep{episode}.pkl"),
        )

    time.sleep(1)

env.close()
print("🎉 학습 완료!")
