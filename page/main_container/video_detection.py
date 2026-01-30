#!/usr/bin/env python3
"""
CCTV Araç ve Motosiklet Tespiti - Video İşleme
Eğitilen YOLOv11 modeli ile Test.avi videosunu işler
"""

import cv2
import torch
from ultralytics import YOLO
import os
import time
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 1
        self.objects = {}  # id -> {'centroid': (x, y), 'class': str, 'disappeared': int, 'history': [(x,y)], 'last_side': None, 'counted_dirs': set()}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _compute_distance(self, c1, c2):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) ** 0.5

    def update(self, detections):
        """
        detections: list of dicts [{'centroid': (x,y), 'class': str, 'box': (x1,y1,x2,y2)}]
        Returns mapping: id -> detection dict (augmented with id)
        """
        if len(detections) == 0:
            # increment disappeared count
            to_delete = []
            for object_id, data in self.objects.items():
                data['disappeared'] += 1
                if data['disappeared'] > self.max_disappeared:
                    to_delete.append(object_id)
            for oid in to_delete:
                del self.objects[oid]
            return {}

        # If no existing objects, register all
        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
        else:
            # Try to match detections to existing objects by nearest centroid and same class
            unmatched_detections = set(range(len(detections)))
            # Build cost matrix style greedy match
            matches = []
            for object_id, data in self.objects.items():
                best_idx = None
                best_dist = None
                for i in list(unmatched_detections):
                    if detections[i]['class'] != data['class']:
                        continue
                    dist = self._compute_distance(detections[i]['centroid'], data['centroid'])
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx is not None and best_dist is not None and best_dist <= self.max_distance:
                    matches.append((object_id, best_idx))
                    unmatched_detections.discard(best_idx)
                else:
                    # mark disappeared
                    data['disappeared'] += 1

            # Remove objects that disappeared too long
            to_delete = []
            for object_id, data in self.objects.items():
                if data['disappeared'] > self.max_disappeared:
                    to_delete.append(object_id)
            for oid in to_delete:
                del self.objects[oid]

            # Update matched
            for object_id, det_idx in matches:
                det = detections[det_idx]
                data = self.objects.get(object_id)
                if data is None:
                    continue
                data['centroid'] = det['centroid']
                data['box'] = det['box']
                data['history'].append(det['centroid'])
                # keep reasonable history
                if len(data['history']) > 20:
                    data['history'] = data['history'][-20:]
                data['disappeared'] = 0

            # Register new (unmatched) detections
            for det_idx in unmatched_detections:
                self._register(detections[det_idx])

        # Prepare return mapping
        mapping = {}
        for object_id, data in self.objects.items():
            mapping[object_id] = {
                'id': object_id,
                'centroid': data['centroid'],
                'class': data['class'],
                'box': data.get('box'),
                'history': data['history'],
                'last_side': data.get('last_side'),
                'counted_dirs': data.get('counted_dirs', set())
            }
        return mapping

    def _register(self, det):
        self.objects[self.next_object_id] = {
            'centroid': det['centroid'],
            'class': det['class'],
            'box': det['box'],
            'disappeared': 0,
            'history': [det['centroid']],
            'last_side': None,
            'counted_dirs': set()
        }
        self.next_object_id += 1

def get_screen_size():
    """Ekran çözünürlüğünü al"""
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        # Fallback: Varsayılan değerler
        return 1920, 1080

def resize_frame_to_fit(frame, max_width=None, max_height=None):
    """Frame'i ekrana sığacak şekilde ölçeklendir"""
    if max_width is None or max_height is None:
        max_width, max_height = get_screen_size()
        # Ekranın %90'ını kullan (kenarlarda boşluk bırakmak için)
        max_width = int(max_width * 0.9)
        max_height = int(max_height * 0.9)
    
    frame_height, frame_width = frame.shape[:2]
    
    # Ölçeklendirme oranını hesapla
    width_ratio = max_width / frame_width
    height_ratio = max_height / frame_height
    scale_ratio = min(width_ratio, height_ratio, 1.0)  # 1.0'dan büyük olmamalı (büyütme yok)
    
    # Eğer frame zaten ekrana sığıyorsa, ölçeklendirme yapma
    if scale_ratio >= 1.0:
        return frame, (frame_width, frame_height)
    
    # Yeni boyutları hesapla
    new_width = int(frame_width * scale_ratio)
    new_height = int(frame_height * scale_ratio)
    
    # Frame'i ölçeklendir
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_frame, (new_width, new_height)

def main():
    # GPU kullanılabilirliğini kontrol et
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılan cihaz: {device}")
    
    # Eğitilen modeli yükle
    model_path = 'runs/train/cctv_car_bike_detection/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"Model yüklendi: {model_path}")
    
    # Video dosyasını kontrol et
    video_path = 'Hacılar8.15-8.45.avi'
    if not os.path.exists(video_path):
        print(f"Hata: Video dosyası bulunamadı: {video_path}")
        return
    
    # Video yakalayıcıyı başlat
    cap = cv2.VideoCapture(video_path)
    
    # Video özelliklerini al
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ekran boyutunu al
    screen_width, screen_height = get_screen_size()
    print(f"Ekran çözünürlüğü: {screen_width}x{screen_height}")
    
    print(f"Video bilgileri:")
    print(f"  - Çözünürlük: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Toplam frame: {total_frames}")
    print(f"  - Süre: {total_frames/fps:.2f} saniye")
    
    # Görüntüleme için ölçeklendirme boyutlarını hesapla
    display_width = int(screen_width * 0.9)
    display_height = int(screen_height * 0.9)
    
    # Çıktı video ayarları
    output_path = 'Test_detected.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Çoklu alan (polygon) ayarları
    area_list = []  # [{'name': str, 'points': [(x1,y1), (x2,y2), ...]}]
    current_polygon = []
    area_color = (0, 255, 255)  # Sarı
    area_thickness = 2

    drawing_area = True
    scale_x = 1.0
    scale_y = 1.0

    # Basit takipçi
    tracker = CentroidTracker(max_disappeared=30, max_distance=80)

    # Renk kodları (BGR formatında)
    colors = {
        'Araba': (0, 255, 0),    # Yeşil
        'Kamyon': (0, 165, 255), # Turuncu
        'Otobus': (255, 0, 0)    # Mavi (BGR)
    }

    # Sadece bu sınıfları çiz
    allowed_classes = set(colors.keys())

    # Ekranda gösterilecek isim eşlemesi (etiket)
    display_name_map = {
        'Araba': 'Araba',
        'Kamyon': 'Kamyon',
        'Otobus': 'Otobüs',
    }
    
    # Tek pencere oluştur
    window_name = 'CCTV Arac Tespiti'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # İlk frame'i al
    ret_first, first_frame = cap.read()
    if not ret_first:
        print("Hata: Videodan ilk frame alınamadı.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return

    # Çoklu alan çizimi için mouse callback
    def area_mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, drawing_area, scale_x, scale_y
        if not drawing_area:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_x = int(x / scale_x)
            frame_y = int(y / scale_y)
            current_polygon.append((frame_x, frame_y))
            print(f"Nokta eklendi: ({frame_x},{frame_y}) | Toplam: {len(current_polygon)}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            current_polygon = []
            print("Alan çizimi sıfırlandı (sağ tık)")

    cv2.setMouseCallback(window_name, area_mouse_callback)

    print("\n=== ALAN SEÇİMİ ===")
    print("Her alan için noktaları sol tıkla ekle, sağ tıkla sıfırla.")
    print("Alanı bitirmek için 'n' tuşuna bas. Tüm alanlar bitince 's' veya 'Enter' ile başlat. Çıkmak için 'q'.")

    while drawing_area:
        display_frame, display_size = resize_frame_to_fit(first_frame, display_width, display_height)
        scale_x = display_size[0] / width
        scale_y = display_size[1] / height
        overlay = display_frame.copy()
        # Mevcut çizilen polygonu göster
        if len(current_polygon) > 0:
            for pt in current_polygon:
                cv2.circle(overlay, (int(pt[0]*scale_x), int(pt[1]*scale_y)), 6, area_color, -1)
            if len(current_polygon) > 1:
                cv2.polylines(overlay, [np.array([(int(x*scale_x), int(y*scale_y)) for (x,y) in current_polygon], np.int32)], False, area_color, area_thickness)
        # Önceki alanları göster
        for area in area_list:
            pts = [(int(x*scale_x), int(y*scale_y)) for (x,y) in area['points']]
            if len(pts) > 1:
                cv2.polylines(overlay, [np.array(pts, np.int32)], True, (0, 200, 0), 2)
            if pts:
                cv2.putText(overlay, area['name'], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s') and len(current_polygon) >= 3:
            # Alanı bitir, isim al
            name = input("Alan ismi girin: ")
            area_list.append({'name': name, 'points': current_polygon.copy()})
            print(f"Alan eklendi: {name} - {len(current_polygon)} nokta")
            current_polygon = []
        elif key == 13:
            drawing_area = False
        elif key == ord('q'):
            print("Çıkılıyor.")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return
    print(f"Toplam {len(area_list)} alan kaydedildi.")
    # ...devamında video işleme ve sayma kodu...
    
    # --- ALANLAR ARASI GEÇİŞ SAYMA ---
    def point_in_polygon(point, polygon):
        # Ray casting algoritması
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # Geçiş sayaçları: {(from, to): count}
    area_names = [area['name'] for area in area_list]
    transition_counts = {}
    for a in area_names:
        for b in area_names:
            if a != b:
                transition_counts[(a, b)] = 0

    # Her nesne için son bulunduğu alanı takip et
    last_area_per_object = {}

    pending_first_frame = first_frame.copy()
    frame_count = 0
    start_time = time.time()

    print("\nVideo işleme başlatılıyor...")
    print("Çıkmak için 'q' tuşuna basın")

    while True:
        if pending_first_frame is not None:
            frame = pending_first_frame
            pending_first_frame = None
        else:
            ret, frame = cap.read()
            if not ret:
                break
        frame_count += 1

        # Model ile tespit yap
        results = model.track(
            frame,
            conf=0.3,
            tracker="bytetrack.yaml",
            persist=True
        )

        # Detections to tracker format
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                if class_name not in allowed_classes:
                    continue
                if confidence < 0.5:
                    continue
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    'centroid': (cx, cy),
                    'class': class_name,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        tracks = tracker.update(detections)

        # Her nesne için alan tespiti ve geçiş kontrolü
        for object_id, data in tracks.items():
            class_name = data['class']
            x1, y1, x2, y2 = data['box'] if data.get('box') else (0, 0, 0, 0)
            cx, cy = data['centroid']
            history = data.get('history', [])

            color = colors.get(class_name, (255, 255, 255))

            # Kutu ve ID çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            id_label = f"{display_name_map.get(class_name, class_name)} ID:{object_id}"
            label_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, id_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Merkez ve iz çizgisi
            cv2.circle(frame, (cx, cy), 3, color, -1)
            if len(history) >= 2:
                for i in range(1, len(history)):
                    cv2.line(frame, history[i - 1], history[i], color, 1)

            # Hangi alanda?
            current_area = None
            for area in area_list:
                if point_in_polygon((cx, cy), area['points']):
                    current_area = area['name']
                    break
            prev_area = last_area_per_object.get(object_id)
            if prev_area is not None and current_area is not None and prev_area != current_area:
                # Geçiş oldu
                if (prev_area, current_area) in transition_counts:
                    transition_counts[(prev_area, current_area)] += 1
                    print(f"{object_id} ID'li nesne {prev_area} -> {current_area} geçti. Toplam: {transition_counts[(prev_area, current_area)]}")
            if current_area is not None:
                last_area_per_object[object_id] = current_area

        # Alanları çiz
        for area in area_list:
            pts = np.array(area['points'], np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 2)
            if len(pts) > 0:
                cv2.putText(frame, area['name'], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Sayaç overlay
        cv2.rectangle(frame, (10, 45), (400, 60+25*len(transition_counts)), (0, 0, 0), -1)
        y_offset = 70
        for (from_area, to_area), count in transition_counts.items():
            cv2.putText(frame, f"{from_area} -> {to_area}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 25

        # Frame bilgilerini ekle
        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Çıktı videosuna yaz (orijinal boyutta)
        out.write(frame)

        # İlerleme göster
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_current = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"İlerleme: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                  f"FPS: {fps_current:.1f}")

        # Frame'i ekrana sığacak şekilde ölçeklendir ve göster
        display_frame, display_size = resize_frame_to_fit(frame, display_width, display_height)
        if frame_count == 1:
            cv2.resizeWindow(window_name, display_size[0], display_size[1])
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı tarafından durduruldu")
            break
    
    # Temizlik
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Sonuçları göster
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\n=== Video işleme tamamlandı! ===")
    print(f"İşlenen frame sayısı: {frame_count}")
    print(f"Toplam süre: {total_time:.2f} saniye")
    print(f"Ortalama FPS: {avg_fps:.2f}")
    print(f"Çıktı dosyası: {output_path}")
    print(f"\n=== SAYMA SONUÇLARI ===")
    toplam = 0
    for (from_area, to_area), count in transition_counts.items():
        print(f"{from_area} -> {to_area}: {count}")
        toplam += count
    print(f"Toplam geçiş: {toplam}")
    
    # Dosya boyutunu kontrol et
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Çıktı dosya boyutu: {file_size:.2f} MB")

if __name__ == "__main__":
    main()