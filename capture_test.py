import os
import atexit
import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
import subprocess

# 설정
capture_device = 0
capture_fps = 30
capture_width = 640
capture_height = 480
output_directory = "captured_frames"
max_frames = 30  # 저장할 최대 프레임 수

# 디렉토리 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 카메라 초기화
camera = CSICamera(capture_device=capture_device, 
                   capture_fps=capture_fps, 
                   capture_width=capture_width, 
                   capture_height=capture_height)

# 종료 시 카메라 리소스 해제
atexit.register(camera.cap.release)

frame_count = 0  # 프레임 카운트 초기화

try:
    while frame_count < max_frames:  # 최대 프레임 수에 도달할 때까지 반복
        image = camera.read()  # 카메라에서 이미지 읽기
        
        # 저장할 파일 경로 설정
        output_path = os.path.join(output_directory, f"frame_{frame_count:03d}.jpg")
        
        # 이미지 저장
        cv2.imwrite(output_path, image)
        print(f"Saved frame {frame_count} to {output_path}")
        
        frame_count += 1
        
except KeyboardInterrupt:
    # Ctrl+C로 종료 시
    print("Frame capturing stopped.")

