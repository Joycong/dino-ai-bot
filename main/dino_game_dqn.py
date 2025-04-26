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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
MODEL_DIR = "../models"
MEMORY_DIR = "../memory"
LOG_PATH = "../logs/training_log.csv"
GAMMA = 0.99
EPSILON_DECAY = 0.900
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
BATCH_SIZE = 32
TOTAL_EPISODES = 1000

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# 로그 파일 초기화
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward", "Epsilon", "MaxObstacles"])

# ReplayMemory 생성
replay_memory = ReplayMemory(10000)

class DinoDQNAgent:
    def __init__(self, state_size, action_size, memory):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.good_memory = []
        self.epsilon = 1.0
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, good_ratio=0.8): # 좋은 메모리를 최대 80%까지
        if len(self.memory) < self.batch_size:
            return

        good_batch_size = int(self.batch_size * good_ratio)
        normal_batch_size = self.batch_size - good_batch_size

        good_samples = self.good_memory[-good_batch_size:] if len(self.good_memory) >= good_batch_size else self.good_memory
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, model_path)
        self.memory.save(memory_path)
        print(f"Model and memory saved:\n- {model_path}\n- {memory_path}")

    def load(self, model_path, memory_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            print(f"Model loaded from {model_path} (epsilon: {self.epsilon:.4f})")
        if os.path.exists(memory_path):
            self.memory.load(memory_path)

# 최신 파일 검색

def get_latest_files():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("dqn_model_ep") and f.endswith(".pth")]
    memory_files = [f for f in os.listdir(MEMORY_DIR) if f.startswith("replay_ep") and f.endswith(".pkl")]

    def extract_episode(filename, prefix):
        try:
            return int(filename.replace(prefix, "").split(".")[0])
        except:
            return -1

    model_eps = [extract_episode(f, "dqn_model_ep") for f in model_files]
    memory_eps = [extract_episode(f, "replay_ep") for f in memory_files]

    valid_eps = set(model_eps).intersection(set(memory_eps))
    if not valid_eps:
        return None, None, 1

    latest_ep = max(valid_eps)
    return (
        os.path.join(MODEL_DIR, f"dqn_model_ep{latest_ep}.pth"),
        os.path.join(MEMORY_DIR, f"replay_ep{latest_ep}.pkl"),
        latest_ep + 1
    )

# 실행 준비
LATEST_MODEL_PATH, LATEST_MEMORY_PATH, START_EPISODE = get_latest_files()
env = DinoGameEnv()
agent = DinoDQNAgent(state_size=80*80, action_size=3, memory=replay_memory)

if LATEST_MODEL_PATH and LATEST_MEMORY_PATH:
    agent.load(LATEST_MODEL_PATH, LATEST_MEMORY_PATH)

max_obstacles = 0

# 학습 루프
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
        next_state, reward, done = env.step(action)

        if reward == 1:
            episode_obstacles += 1
        elif reward == -10:
            total_reward += reward
            break

        # 메모리 저장
        agent.memory.add(state, action, reward, next_state, done)

        # 좋은 메모리 판별 기준
        threshold = max(1, max_obstacles - int(max_obstacles / 2))
        if episode_obstacles >= threshold:
            agent.good_memory.append((state, action, reward, next_state, done))

        agent.replay()
        state = next_state
        total_reward += reward
        step_count += 1

        print(f"[스텝 {step_count}] 행동: {action}, 보상: {reward}, 종료 여부: {done}, 현재 ε: {agent.epsilon:.4f}")

    # 누적 보상 계산: 장애물 개수 기반 보정
    if episode_obstacles > 0:
        bonus = episode_obstacles * (episode_obstacles + 1) // 2
        total_reward = -10 + bonus

    print(f"\n💀 공룡 사망! 에피소드 {episode} 종료")
    print(f"✅ 총 보상: {total_reward} | 🌵 넘은 장애물 수: {episode_obstacles} | 📉 현재 탐험률 (epsilon): {agent.epsilon:.4f}")
    print("-" * 50)

    # 탐험률 조정
    if episode_obstacles > 0 and episode_obstacles >= max_obstacles:
        diff = episode_obstacles - max_obstacles
        decay_multiplier = 1.0 + diff * 0.1 if diff > 0 else 0.1
        new_eps = max(MIN_EPSILON, agent.epsilon * (EPSILON_DECAY ** decay_multiplier))

        if episode_obstacles > max_obstacles:
            print(f"🌟 장애물 최대 갱신: {episode_obstacles} → 탐험률 대폭 감소 ({agent.epsilon:.4f} → {new_eps:.4f})")
            max_obstacles = episode_obstacles
        else:
            print(f"🟢 장애물 최대 동일: {episode_obstacles} → 탐험률 소폭 감소 ({agent.epsilon:.4f} → {new_eps:.4f})")

        agent.epsilon = new_eps

    else:
        # 기존에 아무 변화도 없던 상황에서도 미세하게 감소
        new_eps = max(MIN_EPSILON, agent.epsilon * 0.999)
        print(f"⚪️ 조건 미달 → 탐험률 미세 감소 ({agent.epsilon:.4f} → {new_eps:.4f})")
        agent.epsilon = new_eps

    # 로그 저장
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, round(agent.epsilon, 4), max_obstacles])

    if episode % 10 == 0:
        agent.update_target_model()
        model_path = os.path.join(MODEL_DIR, f"dqn_model_ep{episode}.pth")
        memory_path = os.path.join(MEMORY_DIR, f"replay_ep{episode}.pkl")
        agent.save(model_path, memory_path)

    time.sleep(1)

env.close()
print("\n학습 완료!")
