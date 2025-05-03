import os
import pyautogui
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 상대경로 기반 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chrome_driver_path = os.path.join(BASE_DIR, "chromedriver-win64", "chromedriver.exe")
chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe" #크롬 브라우저 경로

options = Options()
options.add_argument("--mute-audio")
options.binary_location = chrome_path

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)


# 브라우저로 공룡 게임 페이지 열기
print("Opening game page...")
driver.get("https://dinorunner.com/ko/")

# 페이지 로딩을 기다리기
time.sleep(1)
print("Game page opened!")

# 페이지 타이틀과 URL 확인
print(f"Current page title: {driver.title}")
print(f"Current page URL: {driver.current_url}")

# 창 크기와 위치를 고정
driver.set_window_size(800, 800)  # 창 크기 설정
driver.set_window_position(0, 0)  # 창 위치 설정
print("Window size and position set.")

# 스크롤을 완전히 비활성화
driver.execute_script("window.scrollTo(0, 0);")  # 스크롤을 맨 위로 고정
driver.execute_script("document.body.style.overflow = 'hidden';")  # 스크롤 막기
print("Scroll disabled.")

# 게임 시작 전 점프키를 눌러서 게임 시작
time.sleep(1)  # 게임이 로드될 때까지 기다리기
pyautogui.press("space")
print("Game started with initial jump.")

# 게임 화면 캡처 및 저장 함수
def capture_screen(timestamp):
    # 저장 폴더 지정
    save_dir = "add_images"
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 자동 생성

    # 게임 화면 캡처
    screenshot = pyautogui.screenshot(region=(0, 400, 800, 200))
    screenshot_np = np.array(screenshot)
    screenshot_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환

    # 저장 경로 지정
    filename = f"game_screen_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    # 이미지 저장
    cv2.imwrite(save_path, screenshot_rgb)
    print(f"Captured {save_path}")

# 10초 대기 후 캡처 시작
time.sleep(10)
print("Capturing starts after 10 seconds...")

# 4분 동안 매초 1번씩 캡처
start_time = time.time()
capture_duration = 240  # 4분 (240초)
while time.time() - start_time < capture_duration:
    timestamp = int(time.time())  # 현재 시간을 timestamp로 사용
    capture_screen(timestamp)
    time.sleep(1)  # 1초마다 캡처

# 브라우저 종료
driver.quit()
print("Captured 4 minutes of game screen. Exiting...")
