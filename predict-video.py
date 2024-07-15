import cv2
from ultralytics import YOLO
import utils

# Inisialisasi model YOLO
model = YOLO("E:/Kuliah smt 4/MBKM/sftybike-model/project/runs/detect/train11/weights/best.pt")


print(model)

print("Model loaded")
print("Press 'q' to quit")
print("Classes list: ", utils.class_names)

# Buka file video atau kamera
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://www.youtube.com/watch?v=GuB7nw647QM')
# cap = cv2.VideoCapture("C:/Users/ASUS/Videos/2024-07-13 23-46-48.mp4")

model (source='C:/Users/ASUS/Videos/2024-07-14 11-59-54.mp4', show=True, box=True, show_conf=True, conf=0.5)

# # Periksa apakah video berhasil dibuka
# if not cap.isOpened():
#     print("Error: Tidak dapat membuka video.")
#     exit()

# # Buat jendela untuk menampilkan hasil
# cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)


# while cap.isOpened():
#     # Tekan 'q' untuk keluar dari loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # Baca frame dari video
#     ret, frame = cap.read()
#     if not ret:
#         print("Video selesai atau terjadi kesalahan.")
#         break

#     results = model.predict(frame)

#     for result in results:
#         for det in result.boxes.data.tolist():
#             if len(det) >= 6:
#                 x1, y1, x2, y2, conf, cls = det[:6]
#                 if conf > 0.2:  # Kurangi threshold kepercayaan jika diperlukan
#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{int(cls)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
#     # Tampilkan frame asli dengan hasil deteksi
#     cv2.imshow("YOLOv8 Object Detection", frame)

# # Rilis sumber video dan tutup semua jendela OpenCV
# cap.release()
# cv2.destroyAllWindows()
