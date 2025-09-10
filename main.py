import cvzone
import torch
from ultralytics import YOLO
import cv2 as cv
import time

capture = cv.VideoCapture("animals.mp4")
#capture.set(3,640)
#capture.set(4,320)

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model1 = YOLO("runs/detect/train/weights/best.pt")


prev_time = 0
while True:
    isTrue, frame = capture.read()
    frame = cv.resize(frame, (640, 480))
    results = model1(frame,stream=True, verbose=False)
    names = model1.names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            #print(x1,y1,x2,y2)
            cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),1)
            conf = round(float(box.conf[0]),2)
            #print(conf)
            className = names[int(box.cls)]
            cvzone.putTextRect(frame,f'{className} {conf}',
                         (max(0,x1), max(35,y1)), scale=0.5, thickness=1, offset=4)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv.putText(frame, f'FPS: {int(fps)}', (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    cv.imshow("Video", frame)