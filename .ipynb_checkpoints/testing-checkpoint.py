import cv2
import handTracker as hD
import math
import csv
import signal
import sys
import os
import datetime
import time

hand_detector = hD.HandDetector()

# 강제 종료 시그널 핸들러
def signal_handler(sig, frame):
    print("KeyboardInterrupt - Program interrupted")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# 강제 종료 시그널 등록
signal.signal(signal.SIGINT, signal_handler)

# CSV 파일을 저장할 폴더 생성
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 현재 시간을 기반으로 파일 이름 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"{output_folder}/hand_data_{timestamp}.csv"
video_filename = f"{output_folder}/hand_video_{timestamp}.mp4"

# 피처명
feature_names = ["Frame", "Video Time", "Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Pinky Finger", "Angle"]

# CSV 파일에 피처명 저장
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(feature_names)

cap = cv2.VideoCapture(0)  # 웹캠 또는 연결된 카메라를 사용하려면 인덱스를 변경해야 할 수 있습니다.
fps = 30  # 프레임 속도 설정
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # MP4 형식으로 저장

out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

frame_count = 0  # 현재 프레임 번호

while True:
    success, vid = cap.read()
    if not success:
        print("Failed to read video feed.")
        break

    vid = cv2.flip(vid, 1)  # 영상 좌우 반전 (필요에 따라 사용)

    vid = hand_detector.find_hands(vid)
    positions = hand_detector.find_position(vid)

    thumb = None
    index_finger = None
    middle_finger = None
    ring_finger = None
    pinky_finger = None
    angle = None

    if positions is not None:
        for id, cx, cy in positions:
            cv2.putText(vid, f"({cx}, {cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(positions) >= 21:
            thumb = positions[4][1:]
            index_finger = positions[8][1:]
            middle_finger = positions[12][1:]
            ring_finger = positions[16][1:]
            pinky_finger = positions[20][1:]

            # 엄지 손가락과 손목 사이의 각도 계산
            wrist = positions[0]
            angle = math.degrees(math.atan2(thumb[1] - wrist[1], thumb[0] - wrist[0]))

    # 영상 시간 계산
    video_time = time.strftime('%M:%S', time.gmtime(frame_count / fps))

    # CSV 데이터에 추가
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_count, video_time, thumb, index_finger, middle_finger, ring_finger, pinky_finger, angle])

    # 영상 저장
    out.write(vid)
    
    print(f"Frame: {frame_count}, Video Time: {video_time}, CSV file: {csv_filename}, Video file: {video_filename}")

    frame_count += 1

    cv2.imshow('MediaPipe Hands', vid)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
