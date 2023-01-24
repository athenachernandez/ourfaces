import cv2 as cv
from time import sleep

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed ot open.")
    raise SystemError
else:
    print("Camera ready for pictures.")

sleep(1)

for i in range(100):
    _, frame = cap.read()
    cv.imwrite(f'images/athena/img_{i}.jpg', frame)

print("Hit <Enter when your face is not in front of the camera>")
input()

for i in range(100):
    _, frame = cap.read()
    cv.imwrite(f'images/background/img_{i}.jpg', frame)


    