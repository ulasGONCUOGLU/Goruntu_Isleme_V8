import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time


class VideoContainer:
    """Video paneli için video container bileşeni"""
    
    def __init__(self, parent_frame, colors):
        self.parent_frame = parent_frame
        self.colors = colors
        
        # Video değişkenleri
        self.video_capture = None
        self.video_thread = None
        self.is_playing = False
        self.current_frame = None
        
        # UI oluştur
        self.setup_ui()
        
    def setup_ui(self):
        """UI bileşenlerini oluştur"""
        # Video container
        video_container = tk.Frame(self.parent_frame, bg=self.colors['bg_dark'])
        video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video frame (Canvas kullanarak)
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
        
        # Kontrol paneli
        self.create_control_panel()
        
        # Canvas boyut değişimini dinle
        self.video_frame.bind('<Configure>', self.on_canvas_resize)
        
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
            ('🔄 Sıfırla', self.reset_video)
        ]
        
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
                self.video_frame.delete(self.placeholder_text)
                self.show_notification(f'Video yüklendi: {file_path.split("/")[-1]}')
                self.display_first_frame()
            else:
                messagebox.showerror("Hata", "Video dosyası açılamadı!")
                
    def display_first_frame(self):
        """İlk kareyi göster"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.update_video_frame(frame)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
    def play_video(self):
        """Video oynatmayı başlat"""
        if self.video_capture and self.video_capture.isOpened() and not self.is_playing:
            self.is_playing = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.show_notification('Video oynatılıyor')
            
    def video_loop(self):
        """Video oynatma döngüsü"""
        while self.is_playing and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.update_video_frame(frame)
                time.sleep(0.033)  # ~30 FPS
            else:
                self.is_playing = False
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break
                
    def update_video_frame(self, frame):
        """Video karesini canvas'a çiz"""
        # Frame'i RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Canvas boyutlarını al
        canvas_width = self.video_frame.winfo_width()
        canvas_height = self.video_frame.winfo_height()
        
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
        self.video_frame.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Referansı sakla (garbage collection'dan koru)
        self.video_frame.image = photo
        
    def pause_video(self):
        """Video oynatmayı duraklat"""
        if self.is_playing:
            self.is_playing = False
            self.show_notification('Video duraklatıldı')
            
    def stop_video(self):
        """Video oynatmayı durdur"""
        self.is_playing = False
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.display_first_frame()
        self.show_notification('Video durduruldu')
        
    def reset_video(self):
        """Video'yu sıfırla"""
        self.is_playing = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
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
        
    def on_canvas_resize(self, event):
        """Canvas boyutu değiştiğinde"""
        if self.current_frame is not None:
            self.update_video_frame(self.current_frame)
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
        # Root window'a erişim için parent_frame'in root'unu kullan
        root = self.parent_frame.winfo_toplevel()
        
        def reset_status():
            # Widget hala var mı kontrol et
            try:
                if self.status_bar.winfo_exists():
                    self.status_bar.configure(bg=self.colors['bg_dark'])
            except tk.TclError:
                # Widget zaten destroy edilmiş, hiçbir şey yapma
                pass
        
        root.after(2000, reset_status)
        
    def cleanup(self):
        """Temizlik işlemleri"""
        self.is_playing = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

