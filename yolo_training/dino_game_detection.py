import os
import time
import cv2
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import numpy as np
import pyautogui
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 상대경로 기반 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "yolov5", "runs", "train", "exp4", "weights", "best.pt")
chrome_driver_path = os.path.join(BASE_DIR, "chromedriver-win64", "chromedriver.exe")
chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe" #크롬 브라우저 경로

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 크롬 설정
options = Options()
options.add_argument("--mute-audio")
options.binary_location = chrome_path
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)


# Dino 게임 열기
print("Opening game page...")
driver.get("https://dinorunner.com/ko/")

# 페이지 로딩을 기다리기
time.sleep(1)
print("Game page opened!")

# 창 크기와 위치를 고정
driver.set_window_size(800, 800)
driver.set_window_position(0, 0)
print("Window size and position set.")

# 스크롤 비활성화
driver.execute_script("window.scrollTo(0, 0);")
driver.execute_script("document.body.style.overflow = 'hidden';")
print("Scroll disabled.")


# 탐지와 표시 함수
def detect_and_display():
    screenshot = pyautogui.screenshot(region=(0, 400, 800, 200))
    screenshot_np = np.array(screenshot)
    results = model(screenshot_np)

    for detection in results.xyxy[0]:  # 탐지된 각 객체에 대해
        x1, y1, x2, y2, conf, cls = detection
        label = model.names[int(cls)]

        # 탐지된 객체에 따라 색상 지정 (공룡: 초록색, 장애물: 빨간색)
        if label == 'dinosaur':  # 공룡으로 라벨된 객체
            color = (0, 255, 0)  # 초록색
        else:  # 장애물로 라벨된 객체
            color = (255, 0, 0)  # 빨간색

        # 탐지된 객체에 경계 상자와 텍스트 추가
        cv2.rectangle(screenshot_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(screenshot_np, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 이미지 표시
    cv2.imshow('Dino Game Detection', cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR))


# 루프를 통해 게임 플레이와 감지를 연동
try:
    while True:
        detect_and_display()
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'키로 중지
            break
        time.sleep(0.1)

finally:
    cv2.destroyAllWindows()
    driver.quit()
