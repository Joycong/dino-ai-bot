import time
import numpy as np
import pyautogui
import cv2
import torch
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DinoGameEnv:
    def __init__(self):
        chrome_driver_path = "C:/Users/kovin/DinoProject/dino_env/chromedriver-win64/chromedriver.exe"
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
        options = Options()
        options.add_argument("--mute-audio")
        options.binary_location = chrome_path
        service = Service(chrome_driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get("https://dinorunner.com/ko/")
        time.sleep(1)
        self.driver.set_window_size(800, 800)
        self.driver.set_window_position(0, 0)
        self.driver.execute_script("window.scrollTo(0, 0);")
        self.driver.execute_script("document.body.style.overflow = 'hidden';")
        self.game_over_threshold = 0.295

        # YOLOv5 모델 로드
        self.model_path = 'C:/Users/kovin/DinoProject/dino_env/yolov5/runs/train/exp4/weights/best.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        self.dinosaur_label = 0
        self.obstacle_label = 1

        self.current_obstacle = None
        self.game_over_path = "C:/Users/kovin/DinoProject/data/game_over_image.png"

    def reset(self):
        pyautogui.press("space")
        time.sleep(0.1)
        self.current_obstacle = None
        self.last_obstacle_count = 0  # ← 여기에 추가
        time.sleep(0.1)  # 에피소드 시작 시 타임슬립 추가
        return self.get_state()

    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            pyautogui.press("up")
        elif action == 2:
            pyautogui.keyDown("down")
            time.sleep(0.1)
            pyautogui.keyUp("down")
        time.sleep(0.1)
        next_state = self.get_state()
        reward, done = self.get_reward_and_done()
        return next_state, reward, done

    def get_state(self):
        screenshot = self.get_screenshot()
        state = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (80, 80))
        state = state.flatten()
        return state / 255.0

    def get_screenshot(self):
        screenshot = pyautogui.screenshot(region=(0, 400, 800, 200))
        screenshot_np = np.array(screenshot)
        return screenshot_np

    def detect_obstacles(self):
        screenshot = self.get_screenshot()
        results = self.model(screenshot)
        detected_obstacles = []

        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            if label == 'obstacle':
                x1, y1, x2, y2 = map(int, box)
                detected_obstacles.append((x1, y1, x2, y2))

        return detected_obstacles

    def get_dino_position(self):
        screenshot = self.get_screenshot()
        results = self.model(screenshot)

        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            if label == 'dinosaur':
                x1, y1, x2, y2 = map(int, box)
                return (x1 + x2) // 2, (y1 + y2) // 2

        return None

    def get_reward_and_done(self):
        print("\n[보상 함수 시작] --------------------------")

        screenshot = self.get_screenshot()
        game_over_img = cv2.imread(self.game_over_path, cv2.IMREAD_GRAYSCALE)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(screenshot_gray, game_over_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        done = max_val >= self.game_over_threshold

        if done:
            print("[보상] 게임 종료 감지됨 (game over)")
            print("[보상 함수 종료] ----------------------------")
            return -10, True

        dino_position = self.get_dino_position()
        if dino_position is None:
            print("[보상] 공룡 감지 실패로 종료")
            print("[보상 함수 종료] ----------------------------")
            return -10, True

        dino_x = dino_position[0]
        reward = 0
        threshold = 5  # x1이 늘어났다고 판단할 최소 차이값

        detected_obstacles = self.detect_obstacles()
        print(f"감지된 장애물 수: {len(detected_obstacles)}")

        if detected_obstacles:
            # 가장 가까운 장애물 선택 (X1이 가장 작은 것)
            detected_obstacles.sort(key=lambda obs: obs[0])
            nearest_obstacle = detected_obstacles[0]
            obs_x1, _, obs_x2, _ = nearest_obstacle

            print(f"현재 장애물 X2: {obs_x2}, 공룡 X: {dino_x}")

            if self.current_obstacle is not None:
                cur_x1 = self.current_obstacle[0]

                if obs_x1 > cur_x1 + threshold:
                    reward = 1
                    print(f"장애물 넘음 감지 (X1 증가), 보상 +1. 이전 X1: {cur_x1}, 새 X1: {obs_x1}")

                elif obs_x1 < cur_x1:
                    print(f"장애물 갱신됨: X1={obs_x1}, X2={obs_x2}")
            else:
                print(f"새로운 장애물 설정됨: X1={obs_x1}, X2={obs_x2}")

            self.current_obstacle = nearest_obstacle
        else:
            print("장애물 없음")

        print("[보상 함수 종료] ----------------------------")
        return reward, False

    def close(self):
        self.driver.quit()
