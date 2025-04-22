import time
import numpy as np
import pyautogui
import cv2
import torch
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 불필요한 경고 메시지를 숨깁니다
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 현재 파일 기준으로 프로젝트 루트 경로 계산
project_root = os.path.dirname(os.path.abspath(__file__))

# 📦 DinoGameEnv 클래스 정의
# 이 클래스는 공룡 게임을 자동으로 제어하고, 상태를 추출하며, 보상을 계산하는 환경 역할을 합니다.
class DinoGameEnv:
    def __init__(self):
        # ▶ Chrome 드라이버 경로 설정 (상대 경로 사용)
        chrome_driver_path = os.path.join(project_root, "../dino_env/chromedriver-win64/chromedriver.exe")

        # ▶ 크롬 실행 파일 경로 (시스템에 따라 변경 필요)
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"

        # ▶ 크롬 옵션 설정: 소리 끄기 등
        options = Options()
        options.add_argument("--mute-audio")
        options.binary_location = chrome_path
        service = Service(chrome_driver_path)

        # ▶ 웹 드라이버로 게임 실행
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get("https://dinorunner.com/ko/")
        time.sleep(1)
        self.driver.set_window_size(800, 800)
        self.driver.set_window_position(0, 0)
        self.driver.execute_script("window.scrollTo(0, 0);")
        self.driver.execute_script("document.body.style.overflow = 'hidden';")

        # ▶ 게임 오버 감지를 위한 유사도 임계값
        self.game_over_threshold = 0.295

        # ▶ YOLOv5 모델 로드 (장애물, 공룡 인식)
        self.model_path = os.path.join(project_root, "../yolov5/runs/train/exp4/weights/best.pt")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)

        # ▶ 라벨 인덱스 (학습 시 설정한 값에 맞춰야 함)
        self.dinosaur_label = 0
        self.obstacle_label = 1

        self.current_obstacle = None

        # ▶ 게임 오버 이미지 (비교용 템플릿)
        self.game_over_path = os.path.join(project_root, "../data/game_over_image.png")

    # 🔁 게임을 초기화하고 첫 번째 상태를 반환
    def reset(self):
        pyautogui.press("space")  # 게임 시작
        time.sleep(0.1)
        self.current_obstacle = None
        self.last_obstacle_count = 0
        time.sleep(0.1)
        return self.get_state()

    # ⏭️ 에이전트의 행동을 실행하고 다음 상태, 보상, 게임 종료 여부 반환
    def step(self, action):
        if action == 0:
            pass  # 대기
        elif action == 1:
            pyautogui.press("up")  # 점프
        elif action == 2:
            pyautogui.keyDown("down")  # 웅크리기
            time.sleep(0.1)
            pyautogui.keyUp("down")
        time.sleep(0.1)
        next_state = self.get_state()
        reward, done = self.get_reward_and_done()
        return next_state, reward, done

    # 🧠 현재 게임 화면을 상태로 변환 (흑백 + 크기 축소 + 1차원 벡터)
    def get_state(self):
        screenshot = self.get_screenshot()
        state = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (80, 80))
        state = state.flatten()
        return state / 255.0  # 정규화

    # 📸 현재 화면을 캡처하여 NumPy 배열로 반환
    def get_screenshot(self):
        screenshot = pyautogui.screenshot(region=(0, 400, 800, 200))
        screenshot_np = np.array(screenshot)
        return screenshot_np

    # 🚧 YOLO 모델을 통해 장애물 위치 탐지
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

    # 🦖 공룡의 중심 위치를 계산 (탐지된 바운딩 박스를 기준으로)
    def get_dino_position(self):
        screenshot = self.get_screenshot()
        results = self.model(screenshot)

        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            if label == 'dinosaur':
                x1, y1, x2, y2 = map(int, box)
                return (x1 + x2) // 2, (y1 + y2) // 2

        return None  # 공룡을 찾지 못한 경우

    # 🏁 보상 계산 및 게임 종료 여부 판단
    def get_reward_and_done(self):
        print("\n[보상 함수 시작] --------------------------")

        # ▶ 게임 종료 여부 확인 (게임 오버 이미지와 비교)
        screenshot = self.get_screenshot()
        game_over_img = cv2.imread(self.game_over_path, cv2.IMREAD_GRAYSCALE)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(screenshot_gray, game_over_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        done = max_val >= self.game_over_threshold

        if done:
            print("[보상] 게임 종료 감지됨 (game over)")
            print("[보상 함수 종료] ----------------------------")
            return -10, True  # 패널티 부여

        # ▶ 공룡 위치 확인 실패 시 게임 종료
        dino_position = self.get_dino_position()
        if dino_position is None:
            print("[보상] 공룡 감지 실패로 종료")
            print("[보상 함수 종료] ----------------------------")
            return -10, True

        dino_x = dino_position[0]
        reward = 0
        threshold = 5  # 장애물이 갱신되었다고 판단할 최소 거리 차이

        # ▶ 장애물 탐지 및 가장 가까운 장애물 추적
        detected_obstacles = self.detect_obstacles()
        print(f"[보상] 감지된 장애물 수: {len(detected_obstacles)}")

        if detected_obstacles:
            detected_obstacles.sort(key=lambda obs: obs[0])  # X 좌표 기준 정렬
            nearest_obstacle = detected_obstacles[0]
            obs_x1, _, obs_x2, _ = nearest_obstacle

            print(f"[디버깅] 현재 장애물 X2: {obs_x2}, 공룡 X: {dino_x}")

            if self.current_obstacle is not None:
                cur_x1 = self.current_obstacle[0]

                if obs_x1 > cur_x1 + threshold:
                    reward = 1  # 장애물을 넘었다고 판단
                    print(f"[보상] 장애물 넘음 감지 (X1 증가), 보상 +1. 이전 X1: {cur_x1}, 새 X1: {obs_x1}")
                elif obs_x1 < cur_x1:
                    print(f"[보상] 장애물 갱신됨: X1={obs_x1}, X2={obs_x2}")
            else:
                print(f"[보상] 새로운 장애물 설정됨: X1={obs_x1}, X2={obs_x2}")

            self.current_obstacle = nearest_obstacle
        else:
            print("[보상] 장애물 없음")

        print("[보상 함수 종료] ----------------------------")
        return reward, False

    # 🛑 브라우저 종료
    def close(self):
        self.driver.quit()
