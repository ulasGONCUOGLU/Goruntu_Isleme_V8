import tkinter as tk
from tkinter import ttk
import os
import sqlite3

DOSYALAR_DIR = "dosyalar"
DB_PATH = os.path.join(DOSYALAR_DIR, "database.db")


# ──────────────────────────────────────────────────────────────
# DB yardımcıları
# ──────────────────────────────────────────────────────────────

def _get_connection():
    """DB bağlantısı aç, tablo yoksa oluştur."""
    os.makedirs(DOSYALAR_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def db_get(key: str) -> str | None:
    """settings tablosundan değer oku."""
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def db_set(key: str, value: str) -> None:
    """settings tablosuna değer yaz / güncelle."""
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value)
            )
            conn.commit()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# SettingsContainer
# ──────────────────────────────────────────────────────────────

class SettingsContainer:
    """Ayarlar paneli — sol sekme menüsü + sağ içerik alanı."""

    def __init__(self, parent_frame, colors):
        self.parent_frame = parent_frame
        self.colors = colors
        self.model_buttons: dict[str, tk.Button] = {}
        self._selected_model: str | None = None

        self.frame = tk.Frame(parent_frame, bg=colors['bg_dark'])
        self.frame.pack(fill=tk.BOTH, expand=True)

        self._build_ui()

    # ── Ana layout ─────────────────────────────────────────────

    def _build_ui(self):
        # Başlık
        header = tk.Frame(self.frame, bg=self.colors['bg_medium'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="⚙️  Ayarlar",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=20
        ).pack(side=tk.LEFT, pady=15)

        # Gövde
        body = tk.Frame(self.frame, bg=self.colors['bg_dark'])
        body.pack(fill=tk.BOTH, expand=True)

        # Sol sekme menüsü (200 px sabit)
        self.tab_panel = tk.Frame(body, bg=self.colors['bg_medium'], width=200)
        self.tab_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.tab_panel.pack_propagate(False)

        # Sağ içerik
        self.content_area = tk.Frame(body, bg=self.colors['bg_dark'])
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_tab_menu()
        self._show_tab('model')

    # ── Sekme menüsü ───────────────────────────────────────────

    def _build_tab_menu(self):
        tk.Label(
            self.tab_panel,
            text="KATEGORİLER",
            font=('Segoe UI', 8, 'bold'),
            bg=self.colors['bg_medium'],
            fg='#888888',
            padx=15,
            anchor='w'
        ).pack(fill=tk.X, pady=(20, 6))

        # (key, label) — ileride yeni sekmeler buraya eklenir
        tab_defs = [
            ('model', '🤖  Model Seçimi'),
        ]

        self.tab_buttons: dict[str, tk.Button] = {}
        self.tab_frames: dict[str, tk.Frame] = {}

        for key, label in tab_defs:
            btn = tk.Button(
                self.tab_panel,
                text=label,
                font=('Segoe UI', 10),
                bg=self.colors['bg_medium'],
                fg=self.colors['text'],
                relief=tk.FLAT,
                anchor='w',
                padx=15,
                pady=10,
                cursor='hand2',
                command=lambda k=key: self._show_tab(k)
            )
            btn.pack(fill=tk.X)
            self.tab_buttons[key] = btn

            frame = tk.Frame(self.content_area, bg=self.colors['bg_dark'])
            self.tab_frames[key] = frame

        self._build_model_tab(self.tab_frames['model'])

    def _show_tab(self, key: str):
        for f in self.tab_frames.values():
            f.pack_forget()
        for k, b in self.tab_buttons.items():
            b.configure(bg=self.colors['bg_medium'], fg=self.colors['text'])

        self.tab_frames[key].pack(fill=tk.BOTH, expand=True)
        self.tab_buttons[key].configure(
            bg=self.colors['bg_light'], fg='white'
        )

    # ── Model sekmesi ──────────────────────────────────────────

    def _build_model_tab(self, parent: tk.Frame):
        # Üst bar: başlık + yenile
        top_bar = tk.Frame(parent, bg=self.colors['bg_dark'])
        top_bar.pack(fill=tk.X, padx=30, pady=(30, 6))

        tk.Label(
            top_bar,
            text="Nesne Algılama Modeli",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent']
        ).pack(side=tk.LEFT)

        refresh_btn = tk.Button(
            top_bar,
            text="🔄 Yenile",
            font=('Segoe UI', 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=10, pady=4,
            cursor='hand2',
            command=self._refresh_model_list
        )
        refresh_btn.pack(side=tk.RIGHT)
        self._hover(refresh_btn, self.colors['bg_light'], self.colors['accent'])

        # Açıklama
        tk.Label(
            parent,
            text=f"/{DOSYALAR_DIR} klasöründeki .pt dosyaları listelenir.\n"
                 "Seçtiğiniz model video nesne algılamada kullanılır.",
            font=('Segoe UI', 9),
            bg=self.colors['bg_dark'],
            fg='#888888',
            justify=tk.LEFT
        ).pack(anchor='w', padx=30, pady=(0, 16))

        # Scrollable model listesi
        list_wrapper = tk.Frame(parent, bg=self.colors['bg_dark'])
        list_wrapper.pack(fill=tk.BOTH, expand=True, padx=30)

        self._list_canvas = tk.Canvas(
            list_wrapper, bg=self.colors['bg_dark'], highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            list_wrapper, orient='vertical', command=self._list_canvas.yview
        )
        self.model_list_frame = tk.Frame(
            self._list_canvas, bg=self.colors['bg_dark']
        )
        self.model_list_frame.bind(
            '<Configure>',
            lambda e: self._list_canvas.configure(
                scrollregion=self._list_canvas.bbox('all')
            )
        )
        self._list_canvas.create_window(
            (0, 0), window=self.model_list_frame, anchor='nw'
        )
        self._list_canvas.configure(yscrollcommand=scrollbar.set)
        self._list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Alt bar: aktif model etiketi + Uygula butonu
        bottom_bar = tk.Frame(parent, bg=self.colors['bg_medium'], height=55)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_bar.pack_propagate(False)

        self.active_label = tk.Label(
            bottom_bar,
            text="Aktif model: —",
            font=('Segoe UI', 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            padx=20
        )
        self.active_label.pack(side=tk.LEFT, pady=15)

        apply_btn = tk.Button(
            bottom_bar,
            text="✅  Modeli Uygula",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            padx=20, pady=6,
            cursor='hand2',
            command=self._apply_model
        )
        apply_btn.pack(side=tk.RIGHT, padx=20, pady=10)
        self._hover(apply_btn, self.colors['accent'], self.colors['accent_hover'])

        # İlk yükleme
        self._refresh_model_list()

    def _refresh_model_list(self):
        """Klasörü tara, listeyi yeniden oluştur."""
        for w in self.model_list_frame.winfo_children():
            w.destroy()
        self.model_buttons.clear()
        self._selected_model = None

        models = self._scan_models()

        if not models:
            tk.Label(
                self.model_list_frame,
                text=f"/{DOSYALAR_DIR} klasöründe .pt dosyası bulunamadı.",
                font=('Segoe UI', 11),
                bg=self.colors['bg_dark'],
                fg='#888888',
                pady=20
            ).pack(anchor='w', padx=10)
            self.active_label.configure(text="Aktif model: —")
            return

        for name in models:
            self._create_model_row(name)

        # DB'den kaydedilmiş aktif modeli işaretle
        saved = db_get('active_model')
        if saved and saved in self.model_buttons:
            self._select_model(saved)
            self.active_label.configure(text=f"Aktif model: {saved}")
        elif models:
            # Kayıt yoksa ilkini seç (henüz kaydetme)
            self._select_model(models[0])

    def _create_model_row(self, model_name: str):
        """Tek model satırı."""
        row = tk.Frame(self.model_list_frame, bg=self.colors['bg_medium'])
        row.pack(fill=tk.X, pady=4, padx=4)

        btn = tk.Button(
            row,
            text=f"  🧠  {model_name}",
            font=('Segoe UI', 11),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            anchor='w',
            cursor='hand2',
            command=lambda m=model_name: self._select_model(m)
        )
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=5)
        self._hover(btn, self.colors['bg_medium'], self.colors['bg_light'])

        # Dosya boyutu
        tk.Label(
            row,
            text=self._file_size_str(model_name),
            font=('Segoe UI', 9),
            bg=self.colors['bg_medium'],
            fg='#888888',
            padx=15
        ).pack(side=tk.RIGHT, pady=10)

        self.model_buttons[model_name] = btn

    def _select_model(self, model_name: str):
        """Satırı görsel olarak seç (DB'ye henüz yazmaz)."""
        for name, btn in self.model_buttons.items():
            row = btn.master
            is_selected = (name == model_name)
            color = self.colors['accent'] if is_selected else self.colors['bg_medium']
            fg    = 'white'              if is_selected else self.colors['text']
            label_fg = 'white'           if is_selected else '#888888'

            btn.configure(bg=color, fg=fg)
            row.configure(bg=color)
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=color, fg=label_fg)

        self._selected_model = model_name

    def _apply_model(self):
        """Seçili modeli DB'ye kaydet."""
        if not self._selected_model:
            return
        db_set('active_model', self._selected_model)
        self.active_label.configure(text=f"Aktif model: {self._selected_model}")
        self._show_toast(f"✅ Model kaydedildi: {self._selected_model}")

    # ── Yardımcılar ────────────────────────────────────────────

    def _scan_models(self) -> list[str]:
        """DOSYALAR_DIR ve tüm alt klasörlerindeki .pt dosyalarını listele."""
        if not os.path.isdir(DOSYALAR_DIR):
            return []
        
        pt_dosyalari = []
        for kok, klasorler, dosyalar in os.walk(DOSYALAR_DIR):
            for dosya in dosyalar:
                if dosya.endswith('best.pt'):
                    # Tam yolu değil, DOSYALAR_DIR'e göre göreceli yolu ekle
                    tam_yol = os.path.join(kok, dosya)
                    goreceli_yol = os.path.relpath(tam_yol, DOSYALAR_DIR)
                    pt_dosyalari.append(goreceli_yol)
        
        return sorted(pt_dosyalari)

    def _file_size_str(self, model_name: str) -> str:
        path = os.path.join(DOSYALAR_DIR, model_name)
        try:
            size = os.path.getsize(path)
            if size >= 1_073_741_824:
                return f"{size / 1_073_741_824:.1f} GB"
            if size >= 1_048_576:
                return f"{size / 1_048_576:.1f} MB"
            return f"{size / 1024:.1f} KB"
        except OSError:
            return "?"

    def _hover(self, widget: tk.Widget, normal: str, hover: str):
        widget.bind('<Enter>', lambda e: widget.configure(bg=hover))
        widget.bind('<Leave>', lambda e: widget.configure(bg=normal))

    def _show_toast(self, message: str):
        """Alt köşede 2.5 sn görünen bildirim."""
        toast = tk.Label(
            self.frame,
            text=message,
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg='white',
            padx=16, pady=8,
            relief=tk.FLAT
        )
        toast.place(relx=1.0, rely=1.0, anchor='se', x=-20, y=-20)
        self.frame.after(2500, toast.destroy)

    # ── Diğer panellerden erişim ───────────────────────────────

    @staticmethod
    def get_active_model() -> str | None:
        """
        Herhangi bir yerden aktif modeli okumak için:
            from page.settings.main import SettingsContainer
            model = SettingsContainer.get_active_model()
        """
        return db_get('active_model')

    def cleanup(self):
        pass