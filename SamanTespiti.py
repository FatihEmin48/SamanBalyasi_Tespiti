import cv2
import time
from ultralytics import YOLOv10

model = YOLOv10('C:\\YOLOv10\\Yolov10saman\\best.pt')

cap = cv2.VideoCapture("trak.mov")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

saman_balyasi_sayisi = 0
last_detection_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.72)

    current_time = time.time()
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy()
        x_min, y_min, x_max, y_max = bbox
        cls = result.cls.cpu().numpy()
        cls_name = results[0].names[int(cls)]

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)
        cv2.putText(frame, cls_name, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if 800 <= y_max <= 900 and current_time - last_detection_time > 5:
            saman_balyasi_sayisi += 1
            last_detection_time = current_time

    cv2.line(frame, (0, 800), (frame_width, 800), (0, 0, 0), 4)
    cv2.line(frame, (0, 900), (frame_width, 900), (0, 0, 0), 4)

    if saman_balyasi_sayisi > 0:
        cv2.putText(frame, f"{saman_balyasi_sayisi} adet saman balyasi tespit edildi", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    out.write(frame)

    cv2.imshow("YOLOv10 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
