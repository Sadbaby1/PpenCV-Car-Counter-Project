from ultralytics import YOLO
import cv2 as cv
import cvzone # to display all detection
import math
from utils import draw_bbox_conf_cls
import torch

# Check if GPU is ready
device = "cuda" if torch.cuda.is_available() else "cpu"

video_path = "videos/cars.mp4"
# 1. Open vieo
cap = cv.VideoCapture(video_path) 


if not cap.isOpened():
    print("Error: Could not open webcam. Check if connected or used by another app.")
    exit()

# 2. Load weights from weight folder
model =YOLO("/yolo-weights/yolov8n.pt") 

curent_class = ["bus","car"," truck"]
detect = 0
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    # Optional: Run YOLO detection on each frame
    results = model(frame, stream = True, device=device)

    # getting the bbox of each result
    
    for result in  results:
        boxes = result.boxes
        for box in boxes:
            cls =int(box.cls[0])
            if model.names[cls] in curent_class :
                detect += 1
                draw_bbox_conf_cls(box, frame, model)
                cv.putText(frame, str(detect),(40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    cv.imshow("yolo", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
print(detect)
cap.release()
cv.destroyAllWindows()

