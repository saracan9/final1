import cv2
import numpy as np
from ultralytics import YOLO
import math


model = YOLO("best.pt")


cap = cv2.VideoCapture(1)

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    people = []

  
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cls == 0:
            label = "Maskeli"
            color = (0, 255, 0)
        elif cls == 1:
            label = "Maskesiz"
            color = (0, 0, 255)
        else:
            continue

        people.append((cx, cy, (x1, y1, x2, y2)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    too_close = False  

   
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            dist = calculate_distance(people[i][:2], people[j][:2])
            if dist < 100:
                too_close = True
                for p in [people[i], people[j]]:
                    x1, y1, x2, y2 = p[2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Too Close", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    if too_close:
        cv2.putText(frame, "UYARI: Sosyal mesafe ihlali!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    print(f"Karede kişi sayısı: {len(people)}")
    cv2.imshow("Maske ve Sosyal Mesafe Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
