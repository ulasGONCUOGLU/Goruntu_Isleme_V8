# Python Görüntü İşleme Projesi - Video Analiz ve Araç Tespiti

Bu proje, CCTV kamera görüntülerinden araç tespiti yaparak alanlar arası geçiş sayımlarını gerçekleştiren bir Python uygulamasıdır. YOLOv11 modeli kullanılarak araçlar (Araba, Kamyon, Otobüs) tespit edilir ve kullanıcı tanımlı alanlar arasındaki geçişler sayılır.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Çalışma Prensibi](#çalışma-prensibi)
- [Teknik Detaylar](#teknik-detaylar)

## ✨ Özellikler

- **Video Oynatma ve İşleme**: MP4, AVI, MKV gibi formatlarda video dosyalarını yükleme ve oynatma
- **Araç Tespiti**: YOLOv11 modeli ile gerçek zamanlı araç tespiti (Araba, Kamyon, Otobüs)
- **Alan Tanımlama**: Kullanıcı tarafından çoklu polygon alanlar tanımlama
- **Geçiş Sayımı**: Tanımlı alanlar arasındaki araç geçişlerini otomatik sayma
- **Nesne Takibi**: Centroid Tracker algoritması ile nesnelerin takibi
- **Veritabanı Kayıtları**: SQLite veritabanı ile video kayıtları ve geçiş sayımlarını saklama
- **Grafik Gösterimi**: Matplotlib ile geçiş sayımlarının görselleştirilmesi
- **Excel Dışa Aktarma**: Seçili kayıtların Excel formatında dışa aktarılması
- **Modern GUI**: Tkinter tabanlı kullanıcı dostu arayüz

## 🔧 Gereksinimler

### Python Versiyonu
- Python 3.8 veya üzeri

### Gerekli Kütüphaneler

```txt
tkinter (Python ile birlikte gelir)
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.19.0
ultralytics>=8.0.0
torch>=1.9.0
matplotlib>=3.3.0
openpyxl>=3.0.0 (Excel export için opsiyonel)
```

## 📦 Kurulum

1. **Projeyi klonlayın veya indirin**
   ```bash
   git clone <repository-url>
   cd V8
   ```

2. **Virtual environment oluşturun (önerilir)**
   ```bash
   python -m venv venv
   ```

3. **Virtual environment'ı aktifleştirin**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Gerekli paketleri yükleyin**
   ```bash
   pip install opencv-python Pillow numpy ultralytics torch matplotlib openpyxl
   ```

5. **YOLOv11 model dosyasını hazırlayın**
   - Model dosyası `runs/train/cctv_car_bike_detection/weights/best.pt` konumunda olmalıdır
   - Eğer model dosyanız yoksa, önce YOLOv11 ile model eğitimi yapmanız gerekmektedir

## 🚀 Kullanım

### Uygulamayı Başlatma

Ana uygulamayı çalıştırmak için:

```bash
python main.py
```

### Temel Kullanım Adımları

1. **Video Yükleme**
   - Ana Sayfa panelinde "📂 Dosya Seç" butonuna tıklayın
   - İşlemek istediğiniz video dosyasını seçin

2. **Alan Tanımlama**
   - "➕ Ekle" butonuna tıklayarak çizim modunu başlatın
   - Video üzerinde sol tıklayarak alan sınırlarını belirleyin (en az 3 nokta)
   - "✓ Tamamla" butonuna tıklayıp alana bir isim verin
   - İstediğiniz kadar alan ekleyebilirsiniz

3. **Tespit ve Sayım**
   - "🎯 Tespit Aç/Kapa" butonuna tıklayarak tespit modunu aktifleştirin
   - Model otomatik olarak yüklenecektir (ilk kullanımda)
   - "▶️ Oynat" butonuna tıklayarak videoyu başlatın
   - Araçlar tespit edilir ve alanlar arası geçişler sayılır

4. **Sonuçları Görüntüleme**
   - Sağ panelde geçiş sayımları gerçek zamanlı olarak görüntülenir
   - Grafik panelinde ("📊 Grafik" ikonu) tüm kayıtların grafiklerini görebilirsiniz
   - Tablodan bir kayıt seçerek detaylı grafiğini görüntüleyebilirsiniz

5. **Video Kaydetme**
   - Video oynatılırken otomatik olarak kayıt başlar
   - Video durdurulduğunda kayıt için bir isim istenir
   - Kayıtlar `dosyalar/video/` klasörüne kaydedilir
   - Geçiş sayımları veritabanına kaydedilir

6. **Excel Dışa Aktarma**
   - Grafik panelinde bir kayıt seçin
   - "📤 Excel Dışarıya Aktar" butonuna tıklayın
   - Dosya konumunu seçin ve kaydedin

## 📁 Proje Yapısı

```
V8/
├── main.py                          # Ana uygulama giriş noktası
├── page/                            # Sayfa modülleri
│   ├── main_container/              # Ana sayfa container'ları
│   │   ├── video.py                 # Video oynatma ve tespit
│   │   ├── video_detection.py       # Standalone tespit scripti
│   │   └── save.py                  # Video kayıt ve veritabanı işlemleri
│   ├── grafik/                      # Grafik gösterim modülü
│   │   └── main.py                  # Grafik container
│   └── video_container/             # Video container modülü
│       └── video.py
├── dosyalar/                        # Veri klasörleri
│   ├── database.db                  # SQLite veritabanı (otomatik oluşturulur)
│   └── video/                       # Kaydedilen videolar
├── runs/                            # Model eğitim çıktıları
│   └── train/
│       └── cctv_car_bike_detection/
│           └── weights/
│               └── best.pt          # Eğitilmiş YOLOv11 modeli
└── README.md                        # Bu dosya
```

## ⚙️ Çalışma Prensibi

### 1. Video İşleme Akışı

```
Video Yükleme → Frame Okuma → YOLO Tespit → Nesne Takibi → Alan Kontrolü → Geçiş Sayımı
```

### 2. Nesne Tespiti

- **YOLOv11 Modeli**: Her frame'de araçları tespit eder
- **Sınıflar**: Araba, Kamyon, Otobüs
- **Güven Eşiği**: %50 (0.5 confidence)
- **Takip**: ByteTrack algoritması ile nesne takibi

### 3. Nesne Takibi

- **Centroid Tracker**: Her nesnenin merkez noktasını takip eder
- **ID Atama**: Her nesneye benzersiz bir ID atanır
- **Geçmiş Takibi**: Son 20 pozisyon kaydedilir (iz çizgisi için)

### 4. Alan Tanımlama

- **Polygon Çizimi**: Kullanıcı mouse ile polygon çizer
- **Ray Casting Algoritması**: Noktanın polygon içinde olup olmadığını kontrol eder
- **Çoklu Alan**: Birden fazla alan tanımlanabilir

### 5. Geçiş Sayımı

- Her nesne için son bulunduğu alan takip edilir
- Nesne bir alandan başka bir alana geçtiğinde sayaç artırılır
- Geçişler `(from_area, to_area)` çifti olarak kaydedilir

### 6. Veritabanı Yapısı

**video_records tablosu:**
- `id`: Benzersiz kayıt ID'si
- `name`: Video kaydı ismi
- `video_path`: Video dosya yolu
- `created_at`: Oluşturulma tarihi
- `frame_count`: Toplam frame sayısı

**transition_counts tablosu:**
- `id`: Benzersiz ID
- `video_record_id`: Video kaydı referansı
- `from_area`: Başlangıç alanı
- `to_area`: Hedef alan
- `count`: Geçiş sayısı

## 🔬 Teknik Detaylar

### Kullanılan Teknolojiler

- **GUI Framework**: Tkinter
- **Görüntü İşleme**: OpenCV (cv2)
- **Derin Öğrenme**: PyTorch, Ultralytics YOLO
- **Veri Görselleştirme**: Matplotlib
- **Veritabanı**: SQLite3
- **Nesne Takibi**: Centroid Tracker (özel implementasyon)

### Performans Optimizasyonları

- **Threading**: Video oynatma ayrı thread'de çalışır (UI donmaması için)
- **Frame Ölçeklendirme**: Video frame'leri ekrana sığacak şekilde ölçeklenir
- **GPU Desteği**: CUDA kullanılabilirse GPU ile hızlandırma

### Model Eğitimi

Model eğitimi için YOLOv11 kullanılmıştır. Eğitim parametreleri `runs/train/cctv_car_bike_detection/args.yaml` dosyasında bulunabilir.

## 📝 Notlar

- Model dosyası (`best.pt`) projeye dahil değildir, ayrıca sağlanmalıdır
- İlk tespit modu açılışında model yükleme biraz zaman alabilir
- GPU kullanımı performansı önemli ölçüde artırır
- Video kayıtları otomatik olarak `dosyalar/video/` klasörüne kaydedilir

## 🐛 Bilinen Sorunlar

- Model dosyası yoksa tespit modu çalışmaz
- Çok büyük video dosyaları bellek sorunlarına yol açabilir
- Excel export için `openpyxl` paketi gerekir (yoksa CSV olarak kaydedilir)

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👤 Geliştirici

Proje geliştirme sürecinde sorularınız için iletişime geçebilirsiniz.

---

**Not**: Bu README dosyası projenin genel yapısını ve kullanımını açıklamaktadır. Detaylı teknik dokümantasyon için kod içindeki yorumları inceleyebilirsiniz.
