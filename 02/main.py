from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QAction, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QSlider,
    QSizePolicy, QMessageBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas



def pil_to_qpixmap(pil_img: Image.Image, max_side: Optional[int] = None) -> QPixmap:
    img = pil_img
    if max_side:
        w, h = img.size
        m = max(w, h)
        if m > max_side and m > 0:
            s = max_side / m
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimage = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


def compute_hist_rgb(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    r = np.bincount(arr[:, :, 0].ravel(), minlength=256)
    g = np.bincount(arr[:, :, 1].ravel(), minlength=256)
    b = np.bincount(arr[:, :, 2].ravel(), minlength=256)
    return r.astype(np.int64), g.astype(np.int64), b.astype(np.int64)


#  В ручную реализованные фильтры изображений

def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

def brighten_rgb(arr_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Яркость: I' = I * factor (factor∈[0..2])."""
    return _clip_u8(arr_rgb.astype(np.float32) * factor)

def contrast_rgb(arr_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Контраст: I' = 128 + factor * (I - 128)  (factor∈[0..2])."""
    return _clip_u8(128.0 + (arr_rgb.astype(np.float32) - 128.0) * factor)

def invert_rgb(arr_rgb: np.ndarray) -> np.ndarray:
    """Инверсия: I' = 255 - I."""
    return 255 - arr_rgb

def grayscale_mean(arr_rgb: np.ndarray) -> np.ndarray:
    """Ч/Б по среднему: G = (R + G + B) / 3."""
    s = arr_rgb.astype(np.uint16).sum(axis=2)
    return np.round(s / 3.0).astype(np.uint8)

def gaussian_blur_rgb(arr_rgb: np.ndarray, radius: int) -> np.ndarray:
    """Гауссово размытие через библиотеку PIL."""
    if radius <= 0:
        return arr_rgb
    img = Image.fromarray(arr_rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.uint8)

def blur_mix(arr_rgb: np.ndarray, radius: int, f: float) -> np.ndarray:
    """
    Размытиие: I' = (1 - f) * Blur(I, r) + f * I,  f∈[0..1]
    """
    cp = gaussian_blur_rgb(arr_rgb, radius)
    out = (1.0 - f) * cp.astype(np.float32) + f * arr_rgb.astype(np.float32)
    return _clip_u8(out)


class HistCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None):
        fig = Figure(figsize=(3.8, 2.2), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)

    def update_hist(self, hR: np.ndarray, hG: np.ndarray, hB: np.ndarray) -> None:
        ax = self.ax
        ax.clear()

        x = np.arange(256)
        ax.bar(x - 0.3, hR, width=0.3, label="R", color="red",   alpha=0.6)
        ax.bar(x,         hG, width=0.3, label="G", color="green", alpha=0.6)
        ax.bar(x + 0.3,   hB, width=0.3, label="B", color="blue",  alpha=0.6)


        ax.set_xlim(0, 255)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        self.draw_idle()


class ImagePanel(QWidget):
    def __init__(
        self,
        title: str,
        with_slider: bool = False, slider_range: tuple[int, int] = (0, 100),
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.image_label = QLabel("—")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#111; color:#bbb; border-radius:10px;")
        self.image_label.setMinimumSize(220, 220)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hist = HistCanvas(self)

        self.slider: Optional[QSlider] = None
        self.slider_value_label: Optional[QLabel] = None
        if with_slider:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(slider_range[0])
            self.slider.setMaximum(slider_range[1])
            self.slider_value_label = QLabel("0")

        lay = QVBoxLayout(self)
        lay.addWidget(self.title_label)
        lay.addWidget(self.image_label, stretch=1)
        if self.slider is not None:
            lay.addWidget(self.slider); lay.addWidget(self.slider_value_label)
        lay.addWidget(self.hist)

    def set_image(self, pil_img: Image.Image) -> None:
        self.image_label.setPixmap(pil_to_qpixmap(pil_img, max_side=400))
        hR, hG, hB = compute_hist_rgb(pil_img)
        self.hist.update_hist(hR, hG, hB)


class MainWindow(QMainWindow):
    BLUR_MAX_RADIUS = 20

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 2: Фильтры изображений RGB")
        self.resize(1480, 940)

        # состояние
        self.orig_img: Optional[Image.Image] = None
        self.brightness_value = 0   # -100..+100
        self.contrast_value = 0     # -100..+100
        self.blur_radius = 0        # 0..20

        # панели
        self.panel_orig     = ImagePanel("Оригинал")
        self.panel_gray     = ImagePanel("Черно-белое")
        self.panel_invert   = ImagePanel("Инверсия")
        self.panel_bright   = ImagePanel("Яркость",   with_slider=True, slider_range=(-100, 100))
        self.panel_contrast = ImagePanel("Контраст",  with_slider=True, slider_range=(-100, 100))
        self.panel_blur     = ImagePanel(
            "Размытие",
            with_slider=True, slider_range=(0, self.BLUR_MAX_RADIUS),
        )

        # верхняя полоса
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("Выбрать файл…")
        top_bar.addWidget(self.btn_open); top_bar.addStretch(1)

        # раскладка: ряд1 (Оригинал, Ч/Б, Инверсия), ряд2 (Яркость, Контраст, Размытие)
        grid = QGridLayout()
        grid.addWidget(self.panel_orig,     0, 0)
        grid.addWidget(self.panel_gray,     0, 1)
        grid.addWidget(self.panel_invert,   0, 2)
        grid.addWidget(self.panel_bright,   1, 0)
        grid.addWidget(self.panel_contrast, 1, 1)
        grid.addWidget(self.panel_blur,     1, 2)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(top_bar)
        root_layout.addLayout(grid, stretch=1)
        self.setCentralWidget(root)

        # сигналы
        self.btn_open.clicked.connect(self.on_open)
        self.panel_bright.slider.valueChanged.connect(self.on_brightness_changed)
        self.panel_contrast.slider.valueChanged.connect(self.on_contrast_changed)
        self.panel_blur.slider.valueChanged.connect(self.on_blur_radius_changed)

        # хоткей
        act_open = QAction("Open", self); act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.on_open); self.addAction(act_open)


    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.orig_img = img
        self.panel_orig.set_image(img)

        for panel, value, label in (
            (self.panel_bright,   0, "Яркость = +0"),
            (self.panel_contrast, 0, "Контраст = +0"),
        ):
            panel.slider.blockSignals(True)
            panel.slider.setValue(value)
            panel.slider.blockSignals(False)
            panel.slider_value_label.setText(label)

        self.panel_blur.slider.blockSignals(True)
        self.panel_blur.slider.setValue(0)
        self.panel_blur.slider.blockSignals(False)
        self.panel_blur.slider_value_label.setText("Радиус = 0")

        # пересчёт всех панелей по новому оригиналу
        self.update_gray_from_original()
        self.update_invert_from_original()
        self.update_brightness()
        self.update_contrast()
        self.update_blur()

    def on_brightness_changed(self, v: int) -> None:
        self.brightness_value = int(v)
        self.panel_bright.slider_value_label.setText(f"Яркость = {v:+d}")
        self.update_brightness()

    def on_contrast_changed(self, v: int) -> None:
        self.contrast_value = int(v)
        self.panel_contrast.slider_value_label.setText(f"Контраст = {v:+d}")
        self.update_contrast()

    def on_blur_radius_changed(self, v: int) -> None:
        self.blur_radius = int(v)
        self.panel_blur.slider_value_label.setText(f"Радиус = {v}")
        self.update_blur()


    def _factor_brightness(self) -> float:
        return (self.brightness_value + 100) / 100.0

    def _factor_contrast(self) -> float:
        return (self.contrast_value + 100) / 100.0

    def update_brightness(self) -> None:
        if self.orig_img is None: return
        arr = np.asarray(self.orig_img, dtype=np.uint8)
        out = brighten_rgb(arr, self._factor_brightness())
        self.panel_bright.set_image(Image.fromarray(out, mode="RGB"))

    def update_contrast(self) -> None:
        if self.orig_img is None: return
        arr = np.asarray(self.orig_img, dtype=np.uint8)
        out = contrast_rgb(arr, self._factor_contrast())
        self.panel_contrast.set_image(Image.fromarray(out, mode="RGB"))

    def update_blur(self) -> None:
        if self.orig_img is None: return
        arr = np.asarray(self.orig_img, dtype=np.uint8)
        r = max(0, int(self.blur_radius))
        r_max = self.panel_blur.slider.maximum() or self.BLUR_MAX_RADIUS
        f = 1.0 - (r / float(r_max))
        out = blur_mix(arr, r, f)
        self.panel_blur.set_image(Image.fromarray(out, mode="RGB"))

    def update_gray_from_original(self) -> None:
        if self.orig_img is None: return
        arr = np.asarray(self.orig_img, dtype=np.uint8)
        g = grayscale_mean(arr)
        self.panel_gray.set_image(Image.fromarray(g, mode="L"))

    def update_invert_from_original(self) -> None:
        if self.orig_img is None: return
        arr = np.asarray(self.orig_img, dtype=np.uint8)
        inv = invert_rgb(arr)
        self.panel_invert.set_image(Image.fromarray(inv, mode="RGB"))


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
