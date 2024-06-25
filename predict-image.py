import cv2
from ultralytics import YOLO
import numpy as np

# Inisialisasi model YOLO
model = YOLO("runs/detect/train11/weights/best.pt")

# Baca gambar besar
frame = cv2.imread("C:\\Users\\ASUS\\Downloads\\Cuplikan layar 2024-06-25 155238.png")

# Tentukan ukuran grid dan overlap
grid_size = 440
overlap = 220  # 50% overlap

# Boolean untuk mengaktifkan atau menonaktifkan visualisasi grid
visualize_grid = False

# Simpan hasil deteksi dalam list
detections = []

# Iterasi melalui gambar dengan grid dan overlap (horizontal dan vertikal)
for y in range(0, frame.shape[0], grid_size - overlap):
    for x in range(0, frame.shape[1], grid_size - overlap):
        # Crop gambar berdasarkan grid
        crop = frame[y:y + grid_size, x:x + grid_size]

        # Pastikan crop sesuai dengan ukuran grid
        if crop.shape[0] < grid_size or crop.shape[1] < grid_size:
            continue

        # Gambar kotak grid pada gambar asli jika visualize_grid diaktifkan
        if visualize_grid:
            cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (255, 0, 0), 2)  # Warna biru untuk grid

        # Prediksi pada gambar crop
        results = model.predict(crop)

        for result in results:
            for det in result.boxes.data.tolist():
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    if conf > 0.4:  # Kurangi threshold kepercayaan jika diperlukan
                        # Tambahkan deteksi ke dalam list dengan koordinat asli
                        detections.append((x + x1, y + y1, x + x2, y + y2, conf, cls))

# Gambar kotak pembatas pada frame asli
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, f"{int(cls)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Tampilkan frame asli dengan hasil deteksi dan grid jika diaktifkan
cv2.imshow("YOLOv8 Object Detection with Grid", frame)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
