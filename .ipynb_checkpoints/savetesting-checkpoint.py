import cv2
import handTracker as hD
import math
import csv
import os
import datetime

hand_detector = hD.HandDetector()

# CSV 파일을 저장할 폴더 생성
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 현재 시간을 기반으로 파일 이름 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"{output_folder}/hand_data_{timestamp}.csv"

# 피처명
feature_names = ["Frame", "Video Time", "Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Pinky Finger", "Angle"]

# CSV 파일에 피처명 저장
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(feature_names)

# 영상 파일 경로
video_file = "0624_2100_1_1.mp4"

cap = cv2.VideoCapture(video_file)  # 영상 파일 열기
fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 속도 가져오기
frame_count = 0  # 현재 프레임 번호

while True:
    success, vid = cap.read()
    if not success:
        print("Failed to read video frame.")
        break

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
    video_time = str(datetime.timedelta(seconds=frame_count / fps))

    # CSV 데이터에 추가
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_count, video_time, thumb, index_finger, middle_finger, ring_finger, pinky_finger, angle])

    frame_count += 1

    cv2.imshow('MediaPipe Hands', vid)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
