import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import os
from .save import VideoRecorder

# YOLO ve torch import'ları (opsiyonel - yoksa hata vermesin)
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO kütüphanesi bulunamadı. Tespit özellikleri devre dışı.")


class CentroidTracker:
    """Nesne takibi için Centroid Tracker"""
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 1
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _compute_distance(self, c1, c2):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) ** 0.5

    def update(self, detections):
        """Detections listesini güncelle ve ID'leri döndür"""
        if len(detections) == 0:
            to_delete = []
            for object_id, data in self.objects.items():
                data['disappeared'] += 1
                if data['disappeared'] > self.max_disappeared:
                    to_delete.append(object_id)
            for oid in to_delete:
                del self.objects[oid]
            return {}

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
        else:
            unmatched_detections = set(range(len(detections)))
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
                    data['disappeared'] += 1

            to_delete = []
            for object_id, data in self.objects.items():
                if data['disappeared'] > self.max_disappeared:
                    to_delete.append(object_id)
            for oid in to_delete:
                del self.objects[oid]

            for object_id, det_idx in matches:
                det = detections[det_idx]
                data = self.objects.get(object_id)
                if data is None:
                    continue
                data['centroid'] = det['centroid']
                data['box'] = det['box']
                data['history'].append(det['centroid'])
                if len(data['history']) > 20:
                    data['history'] = data['history'][-20:]
                data['disappeared'] = 0

            for det_idx in unmatched_detections:
                self._register(detections[det_idx])

        mapping = {}
        for object_id, data in self.objects.items():
            mapping[object_id] = {
                'id': object_id,
                'centroid': data['centroid'],
                'class': data['class'],
                'box': data.get('box'),
                'history': data['history'],
            }
        return mapping

    def _register(self, det):
        self.objects[self.next_object_id] = {
            'centroid': det['centroid'],
            'class': det['class'],
            'box': det['box'],
            'disappeared': 0,
            'history': [det['centroid']],
        }
        self.next_object_id += 1


def point_in_polygon(point, polygon):
    """Ray casting algoritması ile nokta polygon içinde mi kontrol et"""
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


class MainVideoContainer:
    """Ana Sayfa için video container bileşeni - Görüntü işleme ile"""
    
    def __init__(self, parent_frame, colors):
        self.parent_frame = parent_frame
        self.colors = colors
        
        # Video değişkenleri
        self.video_capture = None
        self.video_thread = None
        self.is_playing = False
        self.current_frame = None
        self.original_frame = None  # Orijinal frame (ölçeklenmemiş)
        self.frame_width = 0
        self.frame_height = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.image_x = 0
        self.image_y = 0
        
        # YOLO model
        self.model = None
        self.tracker = None
        # self.enable_detection = False
        
        # Alan yönetimi
        self.area_list = []  # [{'name': str, 'points': [(x1,y1), ...], 'id': int}]
        self.current_polygon = []
        self.drawing_mode = False
        self.editing_area_id = None
        self.selected_area_id = None
        
        # Geçiş sayımları
        self.transition_counts = {}  # {(from, to): count}
        self.last_area_per_object = {}
        
        # Renk kodları
        self.colors_detection = {
            'Araba': (0, 255, 0),    # Yeşil
            'Kamyon': (0, 165, 255), # Turuncu
            'Otobus': (255, 0, 0)    # Mavi
        }
        self.allowed_classes = set(self.colors_detection.keys())
        
        # Video kayıt sistemi
        self.video_recorder = VideoRecorder()
        self.should_save_on_stop = False  # Kayıt yapılacak mı kontrolü
        
        # UI oluştur
        self.setup_ui()
        
    def setup_ui(self):
        """UI bileşenlerini oluştur"""
        # Ana container (yatay: video %80, bilgi paneli %20)
        main_container = tk.Frame(self.parent_frame, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video container (%80)
        video_container = tk.Frame(main_container, bg=self.colors['bg_dark'])
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Alan yönetimi butonları
        area_control_frame = tk.Frame(video_container, bg=self.colors['bg_medium'])
        area_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_area_frame = tk.Frame(area_control_frame, bg=self.colors['bg_medium'])
        btn_area_frame.pack(pady=10, padx=15)
        
        area_buttons = [
            ('➕ Ekle', self.add_area),
            ('✏️ Düzenle', self.edit_area),
            ('🗑️ Sil', self.delete_area),
            ('✓ Tamamla', self.finish_area)
        ]
        
        for text, command in area_buttons:
            btn = tk.Button(
                btn_area_frame,
                text=text,
                font=('Segoe UI', 9),
                bg=self.colors['accent'],
                fg='white',
                relief=tk.FLAT,
                padx=12,
                pady=6,
                cursor='hand2',
                command=command
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.add_hover_effect(btn, self.colors['accent'], self.colors['accent_hover'])
        
        # Video frame (Canvas)
        self.video_frame = tk.Canvas(
            video_container,
            bg=self.colors['bg_medium'],
            highlightthickness=0
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder metin
        self.placeholder_text = self.video_frame.create_text(
            400, 300,
            text="🎥 Video Oynatıcı\n\nVideo yüklemek için aşağıdaki butonları kullanın",
            font=('Segoe UI', 18),
            fill=self.colors['text'],
            justify=tk.CENTER
        )
        
        # Bilgi paneli (%20 - sağ taraf)
        info_panel = tk.Frame(main_container, bg=self.colors['bg_medium'], width=200)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y)
        info_panel.pack_propagate(False)
        
        # Bilgi paneli başlık
        info_title = tk.Label(
            info_panel,
            text="Geçiş Sayımları",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            pady=15
        )
        info_title.pack()
        
        # Scrollable frame için
        canvas_info = tk.Canvas(info_panel, bg=self.colors['bg_medium'], highlightthickness=0)
        scrollbar_info = tk.Scrollbar(info_panel, orient="vertical", command=canvas_info.yview)
        self.info_content = tk.Frame(canvas_info, bg=self.colors['bg_medium'])
        
        canvas_info.configure(yscrollcommand=scrollbar_info.set)
        canvas_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_info.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas_info.create_window((0, 0), window=self.info_content, anchor="nw")
        self.info_content.bind("<Configure>", lambda e: canvas_info.configure(scrollregion=canvas_info.bbox("all")))
        
        self.info_labels = {}  # Geçiş sayımları için label'lar
        
        # Kontrol paneli
        self.create_control_panel()
        
        # Canvas event'leri
        self.video_frame.bind('<Configure>', self.on_canvas_resize)
        self.video_frame.bind('<Button-1>', self.on_canvas_click)
        self.video_frame.bind('<Motion>', self.on_canvas_motion)
        self.video_frame.bind('<Button-3>', self.on_canvas_right_click)
        
    def create_control_panel(self):
        """Video kontrol panelini oluştur"""
        control_frame = tk.Frame(self.parent_frame, bg=self.colors['bg_medium'])
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Buton container
        btn_container = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        btn_container.pack(pady=15, padx=15)
        
        # Kontrol butonları
        buttons = [
            ('📂 Dosya Seç', self.load_video),
            ('▶️ Oynat', self.play_video),
            ('⏸️ Duraklat', self.pause_video),
            ('⏹️ Durdur', self.stop_video),
            ('🔄 Sıfırla', self.reset_video),
            ('🎥 Video Kaydı Aç/Kapa', self.toggle_video_recording),
        ]

        # Video kayıt butonuna özel referans tut
        self.video_record_button = None

        for text, command in buttons:
            btn = tk.Button(
                btn_container,
                text=text,
                font=('Segoe UI', 10),
                bg=self.colors['accent'],
                fg='white',
                relief=tk.FLAT,
                padx=15,
                pady=8,
                cursor='hand2',
                command=command
            )
            btn.pack(side=tk.LEFT, padx=5)

            if command == self.toggle_video_recording:
                # Bu buton için durumuna göre kırmızı/yeşil renk
                self.video_record_button = btn
                self.update_video_record_button_color()

                btn.bind('<Enter>', self._on_video_record_button_enter)
                btn.bind('<Leave>', self._on_video_record_button_leave)
            else:
                self.add_hover_effect(btn, self.colors['accent'], self.colors['accent_hover'])
        
        # Durum çubuğu
        self.status_bar = tk.Label(
            control_frame,
            text="Durum: Hazır",
            font=('Segoe UI', 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W,
            padx=15,
            pady=10
        )
        self.status_bar.pack(fill=tk.X, padx=15, pady=(0, 15))
        
    def add_hover_effect(self, widget, normal_color, hover_color):
        """Butonlara hover efekti ekle"""
        widget.bind('<Enter>', lambda e: widget.configure(bg=hover_color))
        widget.bind('<Leave>', lambda e: widget.configure(bg=normal_color))
        
    def load_video(self):
        """Video dosyası yükle"""
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(file_path)
            
            if self.video_capture.isOpened():
                self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_frame.delete(self.placeholder_text)
                self.show_notification(f'Video yüklendi: {os.path.basename(file_path)}')
                self.display_first_frame()

                if YOLO_AVAILABLE and self.model is None:
                    model_path = 'runs/train/cctv_car_bike_detection/weights/best.pt'
                    if os.path.exists(model_path):
                        try:
                            self.model = YOLO(model_path)
                            self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)
                            self.show_notification("Model yüklendi")
                        except Exception as e:
                            messagebox.showerror("Hata", f"Model yüklenirken hata: {str(e)}")
                    else:
                        messagebox.showerror("Hata", f"Model dosyası bulunamadı: {model_path}")

            else:
                messagebox.showerror("Hata", "Video dosyası açılamadı!")
                
    def display_first_frame(self):
        """İlk kareyi göster"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.original_frame = frame.copy()
                self.current_frame = frame
                self.update_video_frame(frame)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
    def play_video(self):
        """Video oynatmayı başlat"""
        if self.video_capture and self.video_capture.isOpened() and not self.is_playing:
            self.is_playing = True
            
            # Video kaydı kullanıcı tercihi açıksa başlat
            if self.should_save_on_stop and self.frame_width > 0 and self.frame_height > 0:
                fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
                self.video_recorder.start_recording(
                    self.frame_width,
                    self.frame_height,
                    fps
                )
            
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.show_notification('Video oynatılıyor')
            
    def video_loop(self):
        """Video oynatma döngüsü"""
        while self.is_playing and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.original_frame = frame.copy()
                
                # Tespit aktifse işle
                if self.model:
                    frame = self.process_detection(frame)
                else:
                    # Tespit kapalıysa sadece alanları çiz
                    frame = self.draw_areas_on_frame(frame)
                
                self.current_frame = frame
                
                # Frame'i video kaydına yaz (alanlar ve tespit işaretleri dahil)
                if self.video_recorder.recording:
                    self.video_recorder.write_frame(frame)
                
                self.update_video_frame(frame)
                time.sleep(0.033)  # ~30 FPS
            else:
                # Video bitti
                self.is_playing = False
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Video bittiğinde kayıt/sayım kaydı yap (ana thread'de)
                root = self.parent_frame.winfo_toplevel()
                if self.should_save_on_stop:
                    root.after(0, self._save_recording)
                else:
                    root.after(0, self._save_counts_only)
                break
                
    def process_detection(self, frame):
        """YOLO ile tespit yap ve sayım yap"""
        if not self.model or not self.tracker:
            return frame
        
        # Model ile tespit
        results = self.model.track(
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
                class_name = self.model.names[class_id]
                if class_name not in self.allowed_classes:
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
        
        tracks = self.tracker.update(detections)
        
        # Her nesne için alan tespiti ve geçiş kontrolü
        for object_id, data in tracks.items():
            class_name = data['class']
            x1, y1, x2, y2 = data['box'] if data.get('box') else (0, 0, 0, 0)
            cx, cy = data['centroid']
            history = data.get('history', [])
            
            color = self.colors_detection.get(class_name, (255, 255, 255))
            
            # Kutu ve ID çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            id_label = f"{class_name} ID:{object_id}"
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
            for area in self.area_list:
                if point_in_polygon((cx, cy), area['points']):
                    current_area = area['name']
                    break
            
            prev_area = self.last_area_per_object.get(object_id)
            if prev_area is not None and current_area is not None and prev_area != current_area:
                # Geçiş oldu
                key = (prev_area, current_area)
                if key not in self.transition_counts:
                    self.transition_counts[key] = 0
                self.transition_counts[key] += 1
                self.update_info_panel()
            
            if current_area is not None:
                self.last_area_per_object[object_id] = current_area
        
        # Alanları çiz
        frame = self.draw_areas_on_frame(frame)
        
        return frame
    
    def draw_areas_on_frame(self, frame):
        """Frame üzerine alanları (polygon'ları) çiz"""
        for area in self.area_list:
            pts = np.array(area['points'], np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 2)
            if len(pts) > 0:
                cv2.putText(frame, area['name'], tuple(pts[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
                
    def update_video_frame(self, frame):
        """Video karesini canvas'a çiz"""
        # Frame'i RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Canvas boyutlarını al
        canvas_width = self.video_frame.winfo_width()
        canvas_height = self.video_frame.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Frame boyutlarını hesapla (aspect ratio koru)
        frame_height, frame_width = frame_rgb.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        
        # Resize et
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # PIL Image ve ImageTk'ye çevir
        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=img)
        
        # Canvas'ı temizle ve yeni resmi ekle
        self.video_frame.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Ölçekleme faktörlerini sakla (önce hesapla)
        self.scale_x = new_width / self.frame_width if self.frame_width > 0 else 1.0
        self.scale_y = new_height / self.frame_height if self.frame_height > 0 else 1.0
        self.image_x = x
        self.image_y = y
        
        # Resmi ekle
        self.video_frame.create_image(x, y, anchor=tk.NW, image=photo, tags='image')
        
        # Alanları çiz (Canvas üzerinde)
        self.draw_areas_on_canvas()
        
        # Referansı sakla
        self.video_frame.image = photo
        
    def draw_areas_on_canvas(self):
        """Canvas üzerinde alanları çiz"""
        # Önceki area çizimlerini temizle (image hariç)
        self.video_frame.delete('area')
        self.video_frame.delete('drawing')
        
        for area in self.area_list:
            points = []
            for px, py in area['points']:
                # Orijinal koordinatları canvas koordinatlarına çevir
                canvas_x = self.image_x + int(px * self.scale_x)
                canvas_y = self.image_y + int(py * self.scale_y)
                points.append((canvas_x, canvas_y))
            
            if len(points) > 1:
                # Polygon çiz (kapalı)
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    self.video_frame.create_line(
                        start[0], start[1], end[0], end[1],
                        fill='#00ff00', width=2, tags='area'
                    )
                
                # İsim yaz
                if points:
                    self.video_frame.create_text(
                        points[0][0], points[0][1] - 20,
                        text=area['name'],
                        fill='#ff0000',
                        font=('Arial', 12, 'bold'),
                        tags='area'
                    )
        
        # Çizim modunda mevcut polygon'u göster
        if self.drawing_mode and len(self.current_polygon) > 0:
            canvas_points = []
            for px, py in self.current_polygon:
                canvas_x = self.image_x + int(px * self.scale_x)
                canvas_y = self.image_y + int(py * self.scale_y)
                canvas_points.append((canvas_x, canvas_y))
            
            # Çizgileri çiz
            for i in range(len(canvas_points) - 1):
                start = canvas_points[i]
                end = canvas_points[i + 1]
                self.video_frame.create_line(
                    start[0], start[1], end[0], end[1],
                    fill='#ffff00', width=2, tags='drawing'
                )
            
            # Noktaları göster
            for px, py in canvas_points:
                self.video_frame.create_oval(
                    px - 5, py - 5, px + 5, py + 5,
                    fill='#ffff00', outline='#ffff00', tags='drawing'
                )
    
    def on_canvas_click(self, event):
        """Canvas'a tıklandığında"""
        if not self.drawing_mode or self.original_frame is None:
            return
        
        # Canvas koordinatlarını orijinal frame koordinatlarına çevir
        canvas_x = event.x
        canvas_y = event.y
        
        # Görüntü alanı içinde mi kontrol et
        if (self.image_x <= canvas_x <= self.image_x + int(self.frame_width * self.scale_x) and
            self.image_y <= canvas_y <= self.image_y + int(self.frame_height * self.scale_y)):
            
            # Orijinal koordinatlara çevir
            orig_x = int((canvas_x - self.image_x) / self.scale_x)
            orig_y = int((canvas_y - self.image_y) / self.scale_y)
            
            self.current_polygon.append((orig_x, orig_y))
            self.update_video_frame(self.current_frame if self.current_frame is not None else self.original_frame)
    
    def on_canvas_motion(self, event):
        """Canvas üzerinde mouse hareketi"""
        pass  # İleride hover efekti için kullanılabilir
    
    def on_canvas_right_click(self, event):
        """Canvas'a sağ tıklandığında"""
        if self.drawing_mode:
            self.current_polygon = []
            frame_to_show = self.current_frame if self.current_frame is not None else self.original_frame
            if frame_to_show is not None:
                self.update_video_frame(frame_to_show)
    
    def add_area(self):
        """Yeni alan ekleme modunu başlat"""
        if self.original_frame is None:
            messagebox.showwarning("Uyarı", "Önce bir video yükleyin!")
            return
        
        self.drawing_mode = True
        self.current_polygon = []
        self.editing_area_id = None
        self.show_notification("Çizim modu: Sol tık ile nokta ekleyin, sağ tık ile sıfırlayın")
    
    def finish_area(self):
        """Alan çizimini bitir"""
        if not self.drawing_mode:
            messagebox.showwarning("Uyarı", "Önce çizim modunu başlatın!")
            return
        
        if len(self.current_polygon) < 3:
            messagebox.showwarning("Uyarı", "En az 3 nokta gerekli!")
            return
        
        # İsim al
        name = simpledialog.askstring("Alan İsmi", "Alan ismini girin:")
        if not name:
            return
        
        if self.editing_area_id is not None:
            # Düzenleme modu
            for area in self.area_list:
                if area['id'] == self.editing_area_id:
                    area['points'] = self.current_polygon.copy()
                    area['name'] = name
                    break
            self.editing_area_id = None
        else:
            # Yeni alan ekle
            new_id = max([a['id'] for a in self.area_list], default=0) + 1
            self.area_list.append({
                'id': new_id,
                'name': name,
                'points': self.current_polygon.copy()
            })
            
            # Geçiş sayımlarını güncelle
            self.update_transition_counts()
        
        self.drawing_mode = False
        self.current_polygon = []
        
        # Frame'i güncelle
        frame_to_show = self.current_frame if self.current_frame is not None else self.original_frame
        if frame_to_show is not None:
            self.update_video_frame(frame_to_show)
        
        self.show_notification(f"Alan eklendi: {name}")
    
    def edit_area(self):
        """Alan düzenleme modunu başlat"""
        if not self.area_list:
            messagebox.showwarning("Uyarı", "Düzenlenecek alan yok!")
            return
        
        # Alan seçimi için dialog
        area_names = [f"{a['id']}: {a['name']}" for a in self.area_list]
        selection = simpledialog.askstring(
            "Alan Seç",
            f"Düzenlemek istediğiniz alan ID'sini girin:\n{chr(10).join(area_names)}"
        )
        
        if not selection:
            return
        
        try:
            area_id = int(selection)
            for area in self.area_list:
                if area['id'] == area_id:
                    self.editing_area_id = area_id
                    self.current_polygon = area['points'].copy()
                    self.drawing_mode = True
                    frame_to_show = self.current_frame if self.current_frame is not None else self.original_frame
                    if frame_to_show is not None:
                        self.update_video_frame(frame_to_show)
                    self.show_notification(f"Alan düzenleniyor: {area['name']}")
                    return
            messagebox.showerror("Hata", "Alan bulunamadı!")
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz ID!")
    
    def delete_area(self):
        """Alan sil"""
        if not self.area_list:
            messagebox.showwarning("Uyarı", "Silinecek alan yok!")
            return
        
        area_names = [f"{a['id']}: {a['name']}" for a in self.area_list]
        selection = simpledialog.askstring(
            "Alan Sil",
            f"Silmek istediğiniz alan ID'sini girin:\n{chr(10).join(area_names)}"
        )
        
        if not selection:
            return
        
        try:
            area_id = int(selection)
            self.area_list = [a for a in self.area_list if a['id'] != area_id]
            self.update_transition_counts()
            frame_to_show = self.current_frame if self.current_frame is not None else self.original_frame
            if frame_to_show is not None:
                self.update_video_frame(frame_to_show)
            self.update_info_panel()
            self.show_notification("Alan silindi")
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz ID!")
    
    def update_transition_counts(self):
        """Geçiş sayımları dictionary'sini güncelle"""
        area_names = [area['name'] for area in self.area_list]
        new_counts = {}
        for a in area_names:
            for b in area_names:
                if a != b:
                    key = (a, b)
                    new_counts[key] = self.transition_counts.get(key, 0)
        self.transition_counts = new_counts
        self.update_info_panel()
    
    def update_info_panel(self):
        """Bilgi panelini güncelle"""
        # Mevcut label'ları temizle
        for widget in self.info_content.winfo_children():
            widget.destroy()
        self.info_labels = {}
        
        if not self.transition_counts:
            no_data = tk.Label(
                self.info_content,
                text="Henüz geçiş yok",
                font=('Segoe UI', 10),
                bg=self.colors['bg_medium'],
                fg=self.colors['text']
            )
            no_data.pack(pady=20)
            return
        
        # Geçiş sayımlarını göster
        for (from_area, to_area), count in sorted(self.transition_counts.items()):
            text = f"{from_area} → {to_area}: {count}"
            label = tk.Label(
                self.info_content,
                text=text,
                font=('Segoe UI', 9),
                bg=self.colors['bg_medium'],
                fg='#00ff00',
                anchor=tk.W,
                padx=10,
                pady=5
            )
            label.pack(fill=tk.X, padx=5, pady=2)
            self.info_labels[(from_area, to_area)] = label
    
    def update_video_record_button_color(self):
        """Video kaydı butonunun rengini açık/kapalı durumuna göre ayarla."""
        if not hasattr(self, 'video_record_button') or self.video_record_button is None:
            return

        if self.should_save_on_stop:
            # Açık: yeşil
            bg_color = '#2ecc71'
        else:
            # Kapalı: kırmızı
            bg_color = '#e74c3c'

        self.video_record_button.configure(bg=bg_color, activebackground=bg_color)

    def _on_video_record_button_enter(self, event):
        """Hover sırasında buton rengini biraz koyulaştır."""
        if self.should_save_on_stop:
            hover_color = '#27ae60'
        else:
            hover_color = '#c0392b'
        event.widget.configure(bg=hover_color, activebackground=hover_color)

    def _on_video_record_button_leave(self, event):
        """Hover bitince temel duruma dön."""
        self.update_video_record_button_color()

    def toggle_detection(self):
        """Tespit modunu aç/kapa"""
        if not YOLO_AVAILABLE:
            messagebox.showerror("Hata", "YOLO kütüphanesi bulunamadı!")
            return
        
        if not self.model:
            # Model yükle
            model_path = 'runs/train/cctv_car_bike_detection/weights/best.pt'
            if not os.path.exists(model_path):
                messagebox.showerror("Hata", f"Model dosyası bulunamadı: {model_path}")
                return
            
            try:
                self.model = YOLO(model_path)
                self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)
                self.show_notification("Model yüklendi")
            except Exception as e:
                messagebox.showerror("Hata", f"Model yüklenirken hata: {str(e)}")
                return
        
        self.enable_detection = not self.enable_detection
        status = "açık" if self.enable_detection else "kapalı"
        self.show_notification(f"Tespit modu {status}")
        
        if not self.enable_detection:
            # Sayaçları sıfırla
            self.transition_counts = {}
            self.last_area_per_object = {}
            self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)
            self.update_info_panel()

    def toggle_video_recording(self):
        """Video kaydını aç/kapa"""
        # Oynatma sırasında ayar değiştirmeye izin verme
        if self.is_playing:
            self.show_notification("Önce videoyu durdurun, sonra kayıt ayarını değiştirin")
            return
        
        self.should_save_on_stop = not self.should_save_on_stop
        status = "açık" if self.should_save_on_stop else "kapalı"
        self.show_notification(f"Video kaydı {status}")
        self.update_video_record_button_color()
    
    def pause_video(self):
        """Video oynatmayı duraklat"""
        if self.is_playing:
            self.is_playing = False
            self.show_notification('Video duraklatıldı')
            
    def stop_video(self):
        """Video oynatmayı durdur"""
        was_playing = self.is_playing
        self.is_playing = False
        
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.display_first_frame()
        
        # Kayıt / sayım yapılacaksa kaydet
        if was_playing:
            if self.should_save_on_stop:
                self._save_recording()
            else:
                self._save_counts_only()
        
        self.show_notification('Video durduruldu')
        
    def reset_video(self):
        """Video'yu sıfırla"""
        was_playing = self.is_playing
        self.is_playing = False
        
        # Kayıt / sayım yapılacaksa kaydet
        if was_playing:
            if self.should_save_on_stop:
                self._save_recording()
            else:
                self._save_counts_only()
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.original_frame = None
        self.current_frame = None
        self.frame_width = 0
        self.frame_height = 0
        
        self.video_frame.delete("all")
        self.placeholder_text = self.video_frame.create_text(
            self.video_frame.winfo_width()//2,
            self.video_frame.winfo_height()//2,
            text="🎥 Video Oynatıcı\n\nVideo yüklemek için aşağıdaki butonları kullanın",
            font=('Segoe UI', 18),
            fill=self.colors['text'],
            justify=tk.CENTER
        )
        self.show_notification('Video sıfırlandı')
    
    def _save_recording(self):
        """Video kaydını kaydet"""
        if not self.video_recorder.recording:
            return
        
        # Kullanıcıdan isim iste
        root = self.parent_frame.winfo_toplevel()
        name = simpledialog.askstring(
            "Video Kaydı",
            "Video kaydı için bir isim girin:",
            parent=root
        )
        
        # Geçiş sayımlarını al
        transition_counts = self.transition_counts.copy() if self.transition_counts else None
        
        try:
            # Kayıt işlemini durdur ve kaydet
            result = self.video_recorder.stop_recording(name, transition_counts)
            
            if result:
                self.show_notification(f"Video kaydedildi: {result['name']}")
            else:
                self.show_notification("Video kaydı iptal edildi")
        except Exception as e:
            messagebox.showerror("Hata", f"Video kaydedilirken hata oluştu: {str(e)}")
            self.show_notification("Video kaydı başarısız")
    
    def _save_counts_only(self):
        """Video oluşturmadan sadece geçiş sayımlarını kaydet"""
        if not self.transition_counts:
            return
        
        # Kullanıcıdan isim iste
        root = self.parent_frame.winfo_toplevel()
        name = simpledialog.askstring(
            "Geçiş Sayımları",
            "Geçiş sayımları için bir isim girin:",
            parent=root
        )
        
        if not name:
            self.show_notification("Sayım kaydı iptal edildi")
            return
        
        transition_counts = self.transition_counts.copy()
        
        try:
            record_id = self.video_recorder.save_transition_counts_only(name, transition_counts)
            if record_id:
                self.show_notification(f"Sayım kaydedildi: {name}")
            else:
                self.show_notification("Kaydedilecek geçiş bulunamadı")
        except Exception as e:
            messagebox.showerror("Hata", f"Sayım kaydedilirken hata oluştu: {str(e)}")
            self.show_notification("Sayım kaydı başarısız")
        
    def on_canvas_resize(self, event):
        """Canvas boyutu değiştiğinde"""
        if self.current_frame is not None:
            self.update_video_frame(self.current_frame)
        elif self.original_frame is not None:
            self.update_video_frame(self.original_frame)
        else:
            # Placeholder metnini ortala
            try:
                self.video_frame.coords(
                    self.placeholder_text,
                    event.width//2,
                    event.height//2
                )
            except:
                pass
                
    def show_notification(self, message):
        """Durum çubuğunda bildirim göster"""
        self.status_bar.configure(
            text=f"Durum: {message}",
            bg=self.colors['accent']
        )
        root = self.parent_frame.winfo_toplevel()
        
        def reset_status():
            try:
                if self.status_bar.winfo_exists():
                    self.status_bar.configure(bg=self.colors['bg_dark'])
            except tk.TclError:
                pass
        
        root.after(2000, reset_status)
        
    def cleanup(self):
        """Temizlik işlemleri"""
        self.is_playing = False
        
        # Uygulama kapanırken popup/isim sormadan sadece kaynakları temizle.
        # Eğer kayıt açıksa, kaydı isim vermeden iptal et (geçici dosyayı siler).
        try:
            if self.video_recorder.recording:
                self.video_recorder.stop_recording(name=None, transition_counts=None)
        except Exception:
            # Kapanışta hata yüzünden uygulamayı kilitlemeyelim.
            pass
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Video kayıt sistemini temizle
        self.video_recorder.cleanup()
