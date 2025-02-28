import sys
import os
import cv2
import time
import tempfile
import subprocess
import numpy as np
import pygame
import logging
import traceback
import platform

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, QUrl
from PyQt6.QtGui import QFont, QPalette, QColor, QImage, QPainter, QTextDocument, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QSpinBox, QTextEdit, QLineEdit, QTabWidget, QFormLayout, QGroupBox,
    QMessageBox, QProgressBar, QSlider, QLabel, QCheckBox, QComboBox, QStyleFactory
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

logging.basicConfig(
    filename='save_video_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)
# Функция для чтения изображения с поддержкой Unicode-путей (например, с русскими символами)
def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

# Класс для предобработки (конвертации) видео или GIF в ASCII-арт
class PreprocessingThread(QThread):
    finished = pyqtSignal(list, float)
    progress = pyqtSignal(int, int)

    def __init__(self, video_path, desired_width, ascii_chars, black_white=False):
        super().__init__()
        self.video_path = video_path
        self.desired_width = desired_width
        self.ascii_chars = ascii_chars
        self.black_white = black_white
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit([], 0.0)
            return

        real_fps = cap.get(cv2.CAP_PROP_FPS)
        if not real_fps or real_fps <= 0:
            real_fps = 24.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            total_frames = 0

        ascii_frames = []
        ascii_len = len(self.ascii_chars)
        processed_count = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_h = int(self.desired_width * aspect_ratio * 0.55)
            if new_h < 1:
                new_h = 1

            resized = cv2.resize(frame, (self.desired_width, new_h))
            lines = []
            for row in range(new_h):
                row_buf = []
                for col in range(self.desired_width):
                    b, g, r = resized[row, col]
                    gray = 0.299 * r + 0.587 * g + 0.114 * b
                    idx = int(gray / 255 * (ascii_len - 1))
                    idx = max(0, min(idx, ascii_len - 1))
                    ch = self.ascii_chars[idx]
                    if self.black_white:
                        row_buf.append(ch)
                    else:
                        row_buf.append(f'<span style="color: rgb({r},{g},{b})">{ch}</span>')
                line = "".join(row_buf[:self.desired_width])
                lines.append(line)
            frame_text = "\n".join(lines) if self.black_white else "<br>".join(lines)
            ascii_frames.append(frame_text)
            processed_count += 1
            if total_frames > 0:
                self.progress.emit(processed_count, total_frames)
        cap.release()
        self.finished.emit(ascii_frames, real_fps)

    def stop(self):
        self._run_flag = False

# Главное окно приложения
class AsciiArtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор ASCII-арта")
        self.resize(800, 750)
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(15, 15, 15))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        QApplication.setPalette(dark_palette)

        self.monospace_font = QFont("Courier New", 10)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.image_tab = QWidget()
        self.video_tab = QWidget()
        self.gif_tab = QWidget()

        self.tab_widget.addTab(self.image_tab, "Изображение в ASCII")
        self.tab_widget.addTab(self.video_tab, "Видео в ASCII")
        self.tab_widget.addTab(self.gif_tab, "GIF в ASCII")

        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setStyleSheet("background-color: red; color: white; padding: 5px;")
        self.btn_close.clicked.connect(self.close)
        self.tab_widget.setCornerWidget(self.btn_close, Qt.Corner.TopRightCorner)

        self.init_image_tab()
        self.init_video_tab()
        self.init_gif_tab()

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.play_timer = QTimer()
        self.play_timer.setInterval(15)
        self.play_timer.timeout.connect(self.show_next_frame)

        self.gif_play_timer = QTimer()
        self.gif_play_timer.setInterval(15)
        self.gif_play_timer.timeout.connect(self.show_next_gif_frame)

        self.preproc_thread = None
        self.ascii_frames = []
        self.video_fps = 24.0
        self.video_length = 0
        self.video_start_time = 0
        self.current_frame_index = 0

        self.gif_preproc_thread = None
        self.gif_ascii_frames = []
        self.gif_fps = 24.0
        self.gif_length = 0
        self.gif_start_time = 0
        self.current_gif_frame_index = 0

        self.img_black_white = False
        self.video_black_white = False
        self.gif_black_white = False

        self.current_image_path = ""
        self.current_video_path = ""
        self.current_gif_path = ""

    def closeEvent(self, event):
        self.play_timer.stop()
        self.gif_play_timer.stop()
        if self.preproc_thread:
            self.preproc_thread.stop()
            self.preproc_thread.wait()
        if self.gif_preproc_thread:
            self.gif_preproc_thread.stop()
            self.gif_preproc_thread.wait()
        if self.player:
            self.player.stop()
        pygame.mixer.quit()
        pygame.quit()
        super().closeEvent(event)

    # ----------------------- Изображение в ASCII -----------------------
    def init_image_tab(self):
        layout = QVBoxLayout()
        self.image_tab.setLayout(layout)

        top_layout = QHBoxLayout()
        self.btn_open_img = QPushButton("Открыть изображение")
        self.btn_open_img.clicked.connect(self.open_image)
        top_layout.addWidget(self.btn_open_img)

        width_group = QGroupBox("Ширина")
        width_form = QFormLayout()
        self.img_spin_width = QSpinBox()
        self.img_spin_width.setRange(10, 800)
        self.img_spin_width.setValue(80)
        width_form.addRow("Символов:", self.img_spin_width)
        width_group.setLayout(width_form)
        top_layout.addWidget(width_group)

        charset_group = QGroupBox("Набор символов")
        charset_layout = QVBoxLayout()
        self.img_charset_edit = QLineEdit(".,:;i1tfLCG08@")
        charset_layout.addWidget(self.img_charset_edit)

        self.img_preset_combo = QComboBox()
        self.img_preset_combo.addItem("Default: .,:;i1tfLCG08@", ".,:;i1tfLCG08@")
        self.img_preset_combo.addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
        self.img_preset_combo.addItem("Preset 2", ".:-=+*#%@")
        self.img_preset_combo.addItem("Preset 3", "@%#*+=-:. ")
        self.img_preset_combo.currentIndexChanged.connect(self.update_img_charset)
        charset_layout.addWidget(self.img_preset_combo)
        charset_group.setLayout(charset_layout)
        top_layout.addWidget(charset_group)

        self.img_bw_checkbox = QCheckBox("Черно-белый режим")
        self.img_bw_checkbox.toggled.connect(lambda checked: setattr(self, 'img_black_white', checked))
        top_layout.addWidget(self.img_bw_checkbox)

        self.btn_convert_img = QPushButton("Конвертировать")
        self.btn_convert_img.clicked.connect(self.convert_image_to_ascii)
        top_layout.addWidget(self.btn_convert_img)

        layout.addLayout(top_layout)

        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Масштаб:")
        self.img_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.img_zoom_slider.setRange(5, 30)
        self.img_zoom_slider.setValue(self.monospace_font.pointSize())
        self.img_zoom_slider.valueChanged.connect(self.on_img_zoom_changed)
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.img_zoom_slider)
        layout.addLayout(zoom_layout)

        self.img_ascii_display = QTextEdit()
        self.img_ascii_display.setReadOnly(True)
        self.img_ascii_display.setFont(self.monospace_font)
        self.img_ascii_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.img_ascii_display.setStyleSheet("background-color: black;")
        layout.addWidget(self.img_ascii_display)

        self.progress_image = QProgressBar()
        self.progress_image.setValue(0)
        self.progress_image.setFixedHeight(10)
        layout.addWidget(self.progress_image)

        self.btn_save_img = QPushButton("Сохранить HTML")
        self.btn_save_img.clicked.connect(self.save_html_image)
        layout.addWidget(self.btn_save_img)

        self.btn_save_img_as_pic = QPushButton("Сохранить как изображение")
        self.btn_save_img_as_pic.clicked.connect(self.save_image_as_picture)
        layout.addWidget(self.btn_save_img_as_pic)

    def update_img_charset(self, index):
        preset = self.img_preset_combo.itemData(index)
        self.img_charset_edit.setText(preset)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", 
                                                   "Изображения (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            self.current_image_path = file_name
            self.convert_image_to_ascii()

    def convert_image_to_ascii(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите изображение.")
            return
        img = imread_unicode(self.current_image_path)
        if img is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось открыть изображение.")
            return

        dw = self.img_spin_width.value()
        ascii_chars = self.img_charset_edit.text()
        if not ascii_chars:
            QMessageBox.warning(self, "Ошибка", "Набор символов пуст.")
            return

        h, w = img.shape[:2]
        aspect = h / w
        new_h = max(1, int(dw * aspect * 0.55))
        resized = cv2.resize(img, (dw, new_h))
        lines = []
        length = len(ascii_chars)
        self.progress_image.setValue(0)
        for row in range(new_h):
            row_str = ""
            for col in range(dw):
                b, g, r = resized[row, col]
                gray = 0.299*r + 0.587*g + 0.114*b
                idx = int(gray / 255 * (length - 1))
                idx = max(0, min(idx, length - 1))
                ch = ascii_chars[idx]
                if self.img_black_white:
                    row_str += ch
                else:
                    row_str += f'<span style="color: rgb({r},{g},{b})">{ch}</span>'
            lines.append(row_str)
            progress = int((row + 1) * 100 / new_h)
            self.progress_image.setValue(progress)
            QApplication.processEvents()
        
        if self.img_black_white:
            self.img_ascii_display.setStyleSheet("background-color: black; color: white;")
            self.img_ascii_display.setPlainText("\n".join(lines))
        else:
            self.img_ascii_display.setStyleSheet("background-color: black;")
            self.img_ascii_display.setHtml("<br>".join(lines))

    def save_html_image(self):
        html_code = self.img_ascii_display.toHtml()
        if not html_code.strip():
            QMessageBox.information(self, "Пусто", "Нет ASCII-арта.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить HTML", "", "HTML файлы (*.html)")
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(html_code)
            QMessageBox.information(self, "Успех", f"Сохранено:\n{file_name}")

    def save_image_as_picture(self):
        ascii_text = self.img_ascii_display.toPlainText()
        if not ascii_text.strip():
            QMessageBox.information(self, "Пусто", "Нет ASCII-арта для сохранения.")
            return

        font = self.img_ascii_display.font()
        fm = self.img_ascii_display.fontMetrics()
        lines = ascii_text.split("\n")
        line_count = len(lines)
        max_width = max(fm.horizontalAdvance(line) for line in lines) if lines else 0
        line_height = fm.height()
        img_width = max_width
        img_height = line_height * line_count

        if img_width <= 0 or img_height <= 0:
            QMessageBox.critical(self, "Ошибка", f"Некорректные размеры: {img_width}x{img_height}")
            return

        image = QImage(img_width, img_height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.black)
        painter = QPainter(image)
        painter.setFont(font)
        if self.img_black_white:
            painter.setPen(Qt.GlobalColor.white)
            y = fm.ascent()
            for line in lines:
                painter.drawText(0, y, line)
                y += fm.height()
        else:
            html = self.img_ascii_display.toHtml()
            doc = QTextDocument()
            doc.setHtml(html)
            doc.setDefaultFont(font)
            doc.drawContents(painter)
        painter.end()

        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить как изображение", "",
                                                   "Изображения (*.png *.jpg *.bmp)")
        if file_name:
            if not file_name.lower().endswith(('.png', '.jpg', '.bmp')):
                file_name += '.png'
            if image.save(file_name):
                QMessageBox.information(self, "Успех", f"Изображение сохранено:\n{file_name}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение:\n{file_name}")

    def on_img_zoom_changed(self, value):
        font = self.img_ascii_display.font()
        font.setPointSize(value)
        self.img_ascii_display.setFont(font)

    # ----------------------- Видео в ASCII -----------------------
    
    def init_video_tab(self):
        layout = QVBoxLayout()
        self.video_tab.setLayout(layout)
    
        controls_layout = QHBoxLayout()
        self.btn_open_video = QPushButton("Открыть видео")
        self.btn_open_video.clicked.connect(self.open_video)
        controls_layout.addWidget(self.btn_open_video)
    
        video_width_group = QGroupBox("Ширина")
        video_width_form = QFormLayout()
        self.video_spin_width = QSpinBox()
        self.video_spin_width.setRange(10, 400)
        self.video_spin_width.setValue(78)
        video_width_form.addRow("Символов:", self.video_spin_width)
        video_width_group.setLayout(video_width_form)
        controls_layout.addWidget(video_width_group)
    
        video_charset_group = QGroupBox("Набор символов")
        video_charset_layout = QVBoxLayout()
        self.video_charset_edit = QLineEdit("@%#*+=-:. ")
        video_charset_layout.addWidget(self.video_charset_edit)
    
        self.video_preset_combo = QComboBox()
        self.video_preset_combo.addItem("Default: @%#*+=-:. ", "@%#*+=-:. ")
        self.video_preset_combo.addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
        self.video_preset_combo.addItem("Preset 2", ".:-=+*#%@")
        self.video_preset_combo.addItem("Preset 3", ".,:;i1tfLCG08@")
        self.video_preset_combo.currentIndexChanged.connect(self.update_video_charset)
        video_charset_layout.addWidget(self.video_preset_combo)
        video_charset_group.setLayout(video_charset_layout)
        controls_layout.addWidget(video_charset_group)
    
        self.video_bw_checkbox = QCheckBox("Черно-белый режим")
        self.video_bw_checkbox.toggled.connect(lambda checked: setattr(self, 'video_black_white', checked))
        controls_layout.addWidget(self.video_bw_checkbox)
    
        self.btn_preproc_play = QPushButton("Воспроизвести")
        self.btn_preproc_play.clicked.connect(self.start_preprocessing)
        controls_layout.addWidget(self.btn_preproc_play)
    
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)
    
        layout.addLayout(controls_layout)
    
        self.progress_video = QProgressBar()
        self.progress_video.setValue(0)
        layout.addWidget(self.progress_video)
    
        video_zoom_layout = QHBoxLayout()
        video_zoom_label = QLabel("Масштаб:")
        self.video_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_zoom_slider.setRange(5, 30)
        self.video_zoom_slider.setValue(self.monospace_font.pointSize())
        self.video_zoom_slider.valueChanged.connect(self.on_video_zoom_changed)
        video_zoom_layout.addWidget(video_zoom_label)
        video_zoom_layout.addWidget(self.video_zoom_slider)
        layout.addLayout(video_zoom_layout)
    
        self.video_ascii_display = QTextEdit()
        self.video_ascii_display.setReadOnly(True)
        self.video_ascii_display.setFont(self.monospace_font)
        self.video_ascii_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.video_ascii_display.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_ascii_display)
    
        # Кнопка сохранения видео перенесена сюда
        self.btn_save_video = QPushButton("Сохранить видео")
        self.btn_save_video.clicked.connect(self.save_video_with_audio)
        layout.addWidget(self.btn_save_video)
    
    def update_video_charset(self, index):
        preset = self.video_preset_combo.itemData(index)
        self.video_charset_edit.setText(preset)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "",
                                                   "Видео файлы (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)")
        if file_name:
            self.current_video_path = file_name
            QMessageBox.information(self, "Выбрано видео", self.current_video_path)

    def start_preprocessing(self):
        if not self.current_video_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите видеофайл.")
            return
        w = self.video_spin_width.value()
        chars = self.video_charset_edit.text()
        if not chars:
            QMessageBox.warning(self, "Ошибка", "Набор символов пуст.")
            return

        self.video_ascii_display.clear()
        self.btn_preproc_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_video.setValue(0)

        self.preproc_thread = PreprocessingThread(self.current_video_path, w, chars, self.video_black_white)
        self.preproc_thread.finished.connect(self.on_preprocessing_finished)
        self.preproc_thread.progress.connect(self.on_preprocessing_progress)
        self.preproc_thread.start()

    @pyqtSlot(list, float)
    def on_preprocessing_finished(self, frames, fps):
        self.ascii_frames = frames
        self.video_fps = fps
        self.video_length = len(frames)
        if self.preproc_thread:
            self.preproc_thread.wait()
            self.preproc_thread = None

        if not self.ascii_frames or self.video_length == 0:
            QMessageBox.warning(self, "Ошибка", "Не удалось получить кадры из видео.")
            self.btn_preproc_play.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        self.progress_video.setValue(100)
        self.video_start_time = QDateTime.currentMSecsSinceEpoch()
        self.current_frame_index = 0
        self.video_ascii_display.clear()
        self.play_timer.start()
        self.player.setSource(QUrl.fromLocalFile(self.current_video_path))
        self.player.play()

    @pyqtSlot(int, int)
    def on_preprocessing_progress(self, processed, total):
        if total > 0:
            percentage = int(processed / total * 100)
            self.progress_video.setValue(percentage)
        else:
            self.progress_video.setValue(0)

    def show_next_frame(self):
        current_time = QDateTime.currentMSecsSinceEpoch()
        elapsed = current_time - self.video_start_time
        frame_index = int(elapsed / 1000.0 * self.video_fps)
        if frame_index >= self.video_length:
            self.stop_video()
            return
        if frame_index != self.current_frame_index:
            if self.video_black_white:
                self.video_ascii_display.setStyleSheet("background-color: black; color: white;")
                self.video_ascii_display.setPlainText(self.ascii_frames[frame_index])
            else:
                self.video_ascii_display.setStyleSheet("background-color: black;")
                self.video_ascii_display.setHtml(self.ascii_frames[frame_index])
            self.current_frame_index = frame_index

    def stop_video(self):
        self.play_timer.stop()
        if self.player:
            self.player.stop()
        self.btn_preproc_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.preproc_thread:
            self.preproc_thread.stop()
            self.preproc_thread.wait()
            self.preproc_thread = None

    def on_video_zoom_changed(self, value):
        font = self.video_ascii_display.font()
        font.setPointSize(value)
        self.video_ascii_display.setFont(font)

    def save_video_with_audio(self):
        try:
            if not self.ascii_frames:
                QMessageBox.warning(self, "Ошибка", "Нет обработанных кадров для сохранения видео.")
                return

            file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "", "Видео файлы (*.mp4)")
            if not file_name:
                return
            if not file_name.lower().endswith('.mp4'):
                file_name += '.mp4'

            font = self.video_ascii_display.font()
            fm = QFontMetrics(font)
            char_width = fm.averageCharWidth()
            expected_width = self.video_spin_width.value()
            width = char_width * expected_width
            lines = self.ascii_frames[0].split("\n" if self.video_black_white else "<br>")
            height = fm.height() * len(lines)

            if width <= 0 or height <= 0:
                QMessageBox.critical(self, "Ошибка", f"Некорректные размеры видео: {width}x{height}")
                return

            temp_video = os.path.normpath(os.path.join(tempfile.gettempdir(), "temp_video.avi"))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video, fourcc, self.video_fps, (width, height))
            if not out.isOpened():
                QMessageBox.critical(self, "Ошибка", "Не удалось инициализировать VideoWriter.")
                return

            for i, frame_text in enumerate(self.ascii_frames):
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.black)
                painter = QPainter(image)
                painter.setFont(font)
                if self.video_black_white:
                    painter.setPen(Qt.GlobalColor.white)
                    y = fm.ascent()
                    for line in frame_text.split("\n")[:height // fm.height()]:
                        line = line[:expected_width]
                        painter.drawText(0, y, line)
                        y += fm.height()
                else:
                    doc = QTextDocument()
                    doc.setHtml(frame_text)
                    doc.setDefaultFont(font)
                    doc.setTextWidth(width)
                    doc.drawContents(painter)
                painter.end()

                image = image.convertToFormat(QImage.Format.Format_RGB32)
                ptr = image.bits()
                ptr.setsize(image.sizeInBytes())
                arr = np.array(ptr).reshape(height, width, 4)
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                frame_bgr = np.ascontiguousarray(frame_bgr)
                out.write(frame_bgr)

            out.release()

            # Сначала пытаемся использовать системный FFmpeg
            ffmpeg_path = "ffmpeg"
            try:
                result = subprocess.run([ffmpeg_path, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise FileNotFoundError
            except (FileNotFoundError, OSError):
                # Если системный FFmpeg не найден, используем локальный из ffmpeg/bin
                exe_dir = os.path.dirname(sys.executable)
                ffmpeg_path = os.path.join(exe_dir, "ffmpeg", "bin", "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
                if not os.path.exists(ffmpeg_path):
                    QMessageBox.critical(self, "Ошибка", f"FFmpeg не найден.\n- Системный FFmpeg отсутствует в PATH.\n- Локальный FFmpeg не найден по пути: {ffmpeg_path}")
                    return

            audio_file = os.path.normpath(os.path.join(tempfile.gettempdir(), "temp_audio.mp3"))
            audio_cmd = [ffmpeg_path, '-y', '-i', self.current_video_path, '-vn', '-acodec', 'mp3', audio_file]
            result = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                error_msg = result.stderr.decode()
                QMessageBox.critical(self, "Ошибка", f"Не удалось извлечь аудио:\n{error_msg}")
                return

            merge_cmd = [
                ffmpeg_path, '-y', '-i', temp_video, '-i', audio_file,
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', file_name
            ]
            result = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                error_msg = result.stderr.decode()
                QMessageBox.critical(self, "Ошибка", f"Не удалось объединить видео и аудио:\n{error_msg}")
                return

            try:
                os.remove(temp_video)
                os.remove(audio_file)
            except OSError as e:
                QMessageBox.warning(self, "Предупреждение", f"Не удалось удалить временные файлы: {e}")

            QMessageBox.information(self, "Успех", f"Видео сохранено:\n{file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сохранении видео:\n{str(e)}")
            raise
                                                                              # ----------------------- GIF в ASCII -----------------------
    def init_gif_tab(self):
        layout = QVBoxLayout()
        self.gif_tab.setLayout(layout)

        controls_layout = QHBoxLayout()
        self.btn_open_gif = QPushButton("Открыть GIF")
        self.btn_open_gif.clicked.connect(self.open_gif)
        controls_layout.addWidget(self.btn_open_gif)

        gif_width_group = QGroupBox("Ширина")
        gif_width_form = QFormLayout()
        self.gif_spin_width = QSpinBox()
        self.gif_spin_width.setRange(10, 400)
        self.gif_spin_width.setValue(78)
        gif_width_form.addRow("Символов:", self.gif_spin_width)
        gif_width_group.setLayout(gif_width_form)
        controls_layout.addWidget(gif_width_group)

        gif_charset_group = QGroupBox("Набор символов")
        gif_charset_layout = QVBoxLayout()
        self.gif_charset_edit = QLineEdit("@%#*+=-:. ")
        gif_charset_layout.addWidget(self.gif_charset_edit)

        self.gif_preset_combo = QComboBox()
        self.gif_preset_combo.addItem("Default: @%#*+=-:. ", "@%#*+=-:. ")
        self.gif_preset_combo.addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
        self.gif_preset_combo.addItem("Preset 2", ".:-=+*#%@")
        self.gif_preset_combo.addItem("Preset 3", ".,:;i1tfLCG08@")
        self.gif_preset_combo.currentIndexChanged.connect(self.update_gif_charset)
        gif_charset_layout.addWidget(self.gif_preset_combo)
        gif_charset_group.setLayout(gif_charset_layout)
        controls_layout.addWidget(gif_charset_group)

        self.gif_bw_checkbox = QCheckBox("Черно-белый режим")
        self.gif_bw_checkbox.toggled.connect(lambda checked: setattr(self, 'gif_black_white', checked))
        controls_layout.addWidget(self.gif_bw_checkbox)

        self.btn_preproc_gif = QPushButton("Конвертировать")
        self.btn_preproc_gif.clicked.connect(self.start_preprocessing_gif)
        controls_layout.addWidget(self.btn_preproc_gif)

        self.btn_stop_gif = QPushButton("Остановить")
        self.btn_stop_gif.clicked.connect(self.stop_gif)
        self.btn_stop_gif.setEnabled(False)
        controls_layout.addWidget(self.btn_stop_gif)

        layout.addLayout(controls_layout)

        self.progress_gif = QProgressBar()
        self.progress_gif.setValue(0)
        layout.addWidget(self.progress_gif)

        gif_zoom_layout = QHBoxLayout()
        gif_zoom_label = QLabel("Масштаб:")
        self.gif_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.gif_zoom_slider.setRange(5, 30)
        self.gif_zoom_slider.setValue(self.monospace_font.pointSize())
        self.gif_zoom_slider.valueChanged.connect(self.on_gif_zoom_changed)
        gif_zoom_layout.addWidget(gif_zoom_label)
        gif_zoom_layout.addWidget(self.gif_zoom_slider)
        layout.addLayout(gif_zoom_layout)

        self.gif_ascii_display = QTextEdit()
        self.gif_ascii_display.setReadOnly(True)
        self.gif_ascii_display.setFont(self.monospace_font)
        self.gif_ascii_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.gif_ascii_display.setStyleSheet("background-color: black;")
        layout.addWidget(self.gif_ascii_display)

        self.btn_save_gif = QPushButton("Сохранить GIF")
        self.btn_save_gif.clicked.connect(self.save_gif)
        layout.addWidget(self.btn_save_gif)

    def update_gif_charset(self, index):
        preset = self.gif_preset_combo.itemData(index)
        self.gif_charset_edit.setText(preset)

    def open_gif(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите GIF", "", "GIF файлы (*.gif)")
        if file_name:
            self.current_gif_path = file_name
            QMessageBox.information(self, "Выбран GIF", self.current_gif_path)

    def start_preprocessing_gif(self):
        if not self.current_gif_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите GIF-файл.")
            return
        w = self.gif_spin_width.value()
        chars = self.gif_charset_edit.text()
        if not chars:
            QMessageBox.warning(self, "Ошибка", "Набор символов пуст.")
            return

        self.gif_ascii_display.clear()
        self.btn_preproc_gif.setEnabled(False)
        self.btn_stop_gif.setEnabled(True)
        self.progress_gif.setValue(0)

        self.gif_preproc_thread = PreprocessingThread(self.current_gif_path, w, chars, self.gif_black_white)
        self.gif_preproc_thread.finished.connect(self.on_gif_preprocessing_finished)
        self.gif_preproc_thread.progress.connect(self.on_gif_preprocessing_progress)
        self.gif_preproc_thread.start()

    @pyqtSlot(list, float)
    def on_gif_preprocessing_finished(self, frames, fps):
        self.gif_ascii_frames = frames
        self.gif_fps = fps
        self.gif_length = len(frames)
        if self.gif_preproc_thread:
            self.gif_preproc_thread.wait()
            self.gif_preproc_thread = None

        if not self.gif_ascii_frames or self.gif_length == 0:
            QMessageBox.warning(self, "Ошибка", "Не удалось получить кадры из GIF.")
            self.btn_preproc_gif.setEnabled(True)
            self.btn_stop_gif.setEnabled(False)
            return

        self.progress_gif.setValue(100)
        self.gif_start_time = QDateTime.currentMSecsSinceEpoch()
        self.current_gif_frame_index = 0
        self.gif_ascii_display.clear()
        self.gif_play_timer.start()

    @pyqtSlot(int, int)
    def on_gif_preprocessing_progress(self, processed, total):
        if total > 0:
            percentage = int(processed / total * 100)
            self.progress_gif.setValue(percentage)
        else:
            self.progress_gif.setValue(0)

    def show_next_gif_frame(self):
        if self.gif_length == 0:
            return
        current_time = QDateTime.currentMSecsSinceEpoch()
        elapsed = current_time - self.gif_start_time
        frame_index = int(elapsed / 1000.0 * self.gif_fps) % self.gif_length
        if frame_index != self.current_gif_frame_index:
            if self.gif_black_white:
                self.gif_ascii_display.setStyleSheet("background-color: black; color: white;")
                self.gif_ascii_display.setPlainText(self.gif_ascii_frames[frame_index])
            else:
                self.gif_ascii_display.setStyleSheet("background-color: black;")
                self.gif_ascii_display.setHtml(self.gif_ascii_frames[frame_index])
            self.current_gif_frame_index = frame_index

    def stop_gif(self):
        self.gif_play_timer.stop()
        self.btn_preproc_gif.setEnabled(True)
        self.btn_stop_gif.setEnabled(False)
        if self.gif_preproc_thread:
            self.gif_preproc_thread.stop()
            self.gif_preproc_thread.wait()
            self.gif_preproc_thread = None

    def on_gif_zoom_changed(self, value):
        font = self.gif_ascii_display.font()
        font.setPointSize(value)
        self.gif_ascii_display.setFont(font)

    def save_gif(self):
        try:
            if not self.gif_ascii_frames:
                QMessageBox.warning(self, "Ошибка", "Нет обработанных кадров для сохранения GIF.")
                return

            file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить GIF", "", "GIF файлы (*.gif)")
            if not file_name:
                return
            if not file_name.lower().endswith('.gif'):
                file_name += '.gif'

            font = self.gif_ascii_display.font()
            fm = self.gif_ascii_display.fontMetrics()
            expected_width = self.gif_spin_width.value()
            first_frame = self.gif_ascii_frames[0]
            if self.gif_black_white:
                lines = first_frame.split("\n")
            else:
                lines = first_frame.split("<br>")
            line_count = len(lines)
            char_width = fm.averageCharWidth()
            width = char_width * expected_width
            height = fm.height() * line_count

            temp_video = os.path.join(tempfile.gettempdir(), "temp_gif_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, self.gif_fps, (width, height))
            
            for frame_text in self.gif_ascii_frames:
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.black)
                painter = QPainter(image)
                painter.setFont(font)
                if self.gif_black_white:
                    painter.setPen(Qt.GlobalColor.white)
                    y = fm.ascent()
                    for line in frame_text.split("\n")[:line_count]:
                        painter.drawText(0, y, line[:expected_width])
                        y += fm.height()
                else:
                    doc = QTextDocument()
                    doc.setHtml(frame_text)
                    doc.setDefaultFont(font)
                    doc.setTextWidth(width)
                    doc.drawContents(painter)
                painter.end()
                image = image.convertToFormat(QImage.Format.Format_RGB32)
                ptr = image.bits()
                ptr.setsize(image.sizeInBytes())  # Исправлено: byteCount -> sizeInBytes
                arr = np.array(ptr).reshape(height, width, 4)
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                out.write(frame_bgr)
            out.release()

            # Сначала пытаемся использовать системный FFmpeg
            ffmpeg_path = "ffmpeg"
            try:
                result = subprocess.run([ffmpeg_path, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise FileNotFoundError
            except (FileNotFoundError, OSError):
                # Если системный FFmpeg не найден, используем локальный из ffmpeg/bin
                exe_dir = os.path.dirname(sys.executable)
                ffmpeg_path = os.path.join(exe_dir, "ffmpeg", "bin", "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
                if not os.path.exists(ffmpeg_path):
                    QMessageBox.critical(self, "Ошибка", f"FFmpeg не найден.\n- Системный FFmpeg отсутствует в PATH.\n- Локальный FFmpeg не найден по пути: {ffmpeg_path}")
                    return

            cmd = [
                ffmpeg_path, '-y',
                '-i', temp_video,
                '-vf', f'fps={self.gif_fps},scale={width}:-1:flags=lanczos',
                file_name
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(temp_video)
            if result.returncode == 0:
                QMessageBox.information(self, "Успех", f"GIF сохранён:\n{file_name}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить GIF.\n{result.stderr.decode()}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сохранении GIF:\n{str(e)}")
            raise

if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()
    app = QApplication(sys.argv)
    window = AsciiArtApp()
    window.show()
    sys.exit(app.exec())
