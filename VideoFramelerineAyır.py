import cv2
import os

video_klasoru = "videolar"
frame_klasoru = "frames"

if not os.path.exists(frame_klasoru):
    os.makedirs(frame_klasoru)

for video_dosyasi in os.listdir(video_klasoru):
    if video_dosyasi.endswith((".mp4", ".avi", ".mov")):
        video_yolu = os.path.join(video_klasoru, video_dosyasi)
        video = cv2.VideoCapture(video_yolu)

        if video.isOpened():
            frame_sayaci = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame_adi = video_dosyasi.split('.')[0] + "_frame_" + str(frame_sayaci) + ".jpg"
                frame_yolu = os.path.join(frame_klasoru, frame_adi)
                cv2.imwrite(frame_yolu, frame)
                frame_sayaci += 1

            video.release()

print("Tüm videolar karelere ayrıldı ve kaydedildi")
