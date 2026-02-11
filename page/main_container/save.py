import sqlite3
import os
import cv2
from datetime import datetime


class VideoRecorder:
    """Video kayıt ve veritabanı işlemleri"""
    
    def __init__(self):
        self.video_writer = None
        self.recording = False
        self.recorded_frames = []
        self.db_path = 'dosyalar/database.db'
        self.video_dir = 'dosyalar/video'
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Gerekli klasörleri oluştur"""
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Video kayıtları tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frame_count INTEGER,
                duration REAL
            )
        ''')
        
        # Geçiş sayımları tablosu (video ile ilişkili)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transition_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_record_id INTEGER,
                from_area TEXT NOT NULL,
                to_area TEXT NOT NULL,
                count INTEGER NOT NULL,
                FOREIGN KEY (video_record_id) REFERENCES video_records(id)
            )
        ''')
        
        # Not: Sadece sayım kayıtları da mevcut şema üzerinden (video_records + transition_counts)
        # tutulur. Bu yüzden ayrı bir tabloya ihtiyaç yoktur.
        
        conn.commit()
        conn.close()
    
    def start_recording(self, frame_width, frame_height, fps=30):
        """Video kaydını başlat"""
        if self.recording:
            return False
        
        self.recording = True
        self.frame_count = 0
        
        # VideoWriter oluştur (geçici dosya)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(self.video_dir, f"temp_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            temp_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not self.video_writer.isOpened():
            self.recording = False
            self.video_writer = None
            return False
        
        self.temp_video_path = temp_path
        return True
    
    def write_frame(self, frame):
        """Frame'i video'ya yaz"""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1
    
    def stop_recording(self, name=None, transition_counts=None):
        """Video kaydını durdur ve kaydet
        
        Args:
            name: Video kaydı için isim (None ise geçici dosya silinir)
            transition_counts: Geçiş sayımları dictionary'si
        
        Returns:
            dict: Kayıt bilgileri veya None
        """
        if not self.recording:
            return None
        
        self.recording = False
        
        # VideoWriter'ı kapat
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if not name:
            # İsim verilmezse geçici dosyayı sil
            if hasattr(self, 'temp_video_path') and os.path.exists(self.temp_video_path):
                os.remove(self.temp_video_path)
            return None
        
        try:
            # Dosya adını oluştur
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{safe_name}_{timestamp}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)
            
            # Geçici dosyayı yeniden adlandır
            if hasattr(self, 'temp_video_path') and os.path.exists(self.temp_video_path):
                if os.path.exists(video_path):
                    os.remove(video_path)
                os.rename(self.temp_video_path, video_path)
            
            # Veritabanına kaydet
            record_id = self._save_to_database(
                name,
                video_path,
                self.frame_count,
                transition_counts
            )
            
            return {
                'id': record_id,
                'name': name,
                'video_path': video_path,
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            raise Exception(f"Kayıt sırasında hata oluştu: {str(e)}")
    
    def _save_to_database(self, name, video_path, frame_count, transition_counts=None):
        """Veritabanına kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Video kaydını ekle
            cursor.execute('''
                INSERT INTO video_records (name, video_path, frame_count)
                VALUES (?, ?, ?)
            ''', (name, video_path, frame_count))
            
            record_id = cursor.lastrowid
            
            # Geçiş sayımlarını ekle
            if transition_counts:
                for (from_area, to_area), count in transition_counts.items():
                    if count > 0:  # Sadece 0'dan büyük sayımları kaydet
                        cursor.execute('''
                            INSERT INTO transition_counts (video_record_id, from_area, to_area, count)
                            VALUES (?, ?, ?, ?)
                        ''', (record_id, from_area, to_area, count))
            
            conn.commit()
            return record_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_all_records(self):
        """Tüm video kayıtlarını getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, video_path, created_at, frame_count
            FROM video_records
            ORDER BY created_at DESC
        ''')
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def get_transition_counts(self, video_record_id):
        """Belirli bir video kaydının geçiş sayımlarını getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT from_area, to_area, count
            FROM transition_counts
            WHERE video_record_id = ?
        ''', (video_record_id,))
        
        counts = cursor.fetchall()
        conn.close()
        
        return counts
    
    def cleanup(self):
        """Temizlik işlemleri"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False

    def save_transition_counts_only(self, name, transition_counts):
        """Video dosyası oluşturmadan sadece geçiş sayımlarını veritabanına kaydet.
        Kayıtlar `video_records` + `transition_counts` tablolarına yazılır.
        
        Args:
            name: Kayıt ismi
            transition_counts: {(from_area, to_area): count} sözlüğü
        
        Returns:
            int: Oluşturulan oturum ID'si veya None
        """
        if not transition_counts:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Video olmadan bir "kayıt başlığı" oluştur (video_path zorunlu olduğu için placeholder)
            cursor.execute('''
                INSERT INTO video_records (name, video_path, frame_count)
                VALUES (?, ?, ?)
            ''', (name, '', 0))
            
            record_id = cursor.lastrowid
            
            # Geçiş sayımlarını ekle
            for (from_area, to_area), count in transition_counts.items():
                if count > 0:
                    cursor.execute('''
                        INSERT INTO transition_counts (video_record_id, from_area, to_area, count)
                        VALUES (?, ?, ?, ?)
                    ''', (record_id, from_area, to_area, count))
            
            conn.commit()
            return record_id
        
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
