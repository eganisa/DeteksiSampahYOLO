from ultralytics import YOLO
import cv2
import time

model = YOLO("model/V3_YOLOv8s.pt")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise ("No Camera")

while True:
    ret, image = cam.read()
    if not ret:
        break

    _time_mulai = time.time()
    result = model.predict(image, show=True)

    print("waktu", time.time() - _time_mulai)
    
    # cv2.imshow("image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
