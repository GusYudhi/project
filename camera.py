import cv2
from ultralytics import YOLO
import utils
import time

# Inisialisasi model YOLO
model = YOLO("runs/detect/train11/weights/best.pt", verbose=False)

device = "cpu"

model.to(device)

print("Model loaded")
print("Press 'q' to quit")
print("Classes list: ", utils.class_names)

# Buka file video atau kamera
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("C:\\Users\\ASUS\\Videos\\2024-06-25 09-51-33.mp4")

# Periksa apakah video berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()

frame_skip = 40
frame_count = 0

# Boolean untuk mengaktifkan atau menonaktifkan visualisasi grid
visualize_grid = True

ret, frame = cap.read()

scale_factor = 6 # sesuaikan dengan keadaan kamera

# Tentukan ukuran grid dan overlap
grid_size = int(frame.shape[1] / scale_factor)
overlap = int(grid_size / 2) 

print(f"grid_size: {grid_size}, overlap: {overlap}")

while cap.isOpened():
    # time sleep
    time.sleep(0.5)
    # Baca frame dari video
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau terjadi kesalahan.")
        break

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
                        if conf > 0.3:  # Kurangi threshold kepercayaan jika diperlukan
                            # Tambahkan deteksi ke dalam list dengan koordinat asli
                            detections.append((x + x1, y + y1, x + x2, y + y2, conf, cls))

