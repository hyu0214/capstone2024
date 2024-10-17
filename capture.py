import os
import atexit
import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
import subprocess
import shutil  # shutil 추가

# 설정
capture_device = 0
capture_fps = 10
capture_width = 640
capture_height = 480
output_directory = "/home/ircvlab/HYU-2024-Embedded/code/captured_frames"
feature_points_directory = "/home/ircvlab/HYU-2024-Embedded/code/feature_points"  # feature_points 경로 추가

# feature_points 디렉토리 초기화: 기존 파일 삭제
if os.path.exists(feature_points_directory):
    for filename in os.listdir(feature_points_directory):
        file_path = os.path.join(feature_points_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 파일 삭제
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 디렉토리 삭제
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# 디렉토리 초기화: captured_frames 디렉토리의 기존 파일 삭제
if os.path.exists(output_directory):
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 파일 삭제
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 디렉토리 삭제
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
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
    while True:  # 무한 루프
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

# C++ 프로그램 실행 (필요시)
cpp_program_command = "/home/ircvlab/HYU-2024-Embedded/code/build_1/run_spherical_sfm_realtime -intrinsics /home/ircvlab/HYU-2024-Embedded/code/intrinsics.txt"

# C++ 프로그램 실행
subprocess.run(cpp_program_command, shell=True)
