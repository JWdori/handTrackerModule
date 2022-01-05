import cv2
import handTracker as hT

cap = cv2.VideoCapture("test.mp4")
tr = hT.HandDetector()

while cap.isOpened():
    success, vid = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    vid = tr.find_hands(vid)
    if tr.find_position(vid) != None:
        print(tr.find_position(vid))
    else:
        print("No Hand detected")

    cv2.imshow('MediaPipe Hands', vid)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
