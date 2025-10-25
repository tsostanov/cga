from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QMessageBox,
    QGroupBox,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

IMAGE1_PATH = Path(__file__).parent / "Pictures" / "image1.jpg"
IMAGE2_PATH = Path(__file__).parent / "Pictures" / "image2.jpg"


@dataclass
class ImageStats:
    width: int
    height: int
    total_pixels: int
    # суммы по каналам R, G, B
    sum_r: int
    sum_g: int
    sum_b: int
    # доля канала в общей сумме Σ(R+G+B)
    share_r: float
    share_g: float
    share_b: float
    # гистограммы по каналам
    hist_r: np.ndarray  # shape (256,)
    hist_g: np.ndarray
    hist_b: np.ndarray


# class HistogramDialog(QDialog):
#     """Модальный диалог с гистограммами RGB (0..255)."""

#     def __init__(self, stats: ImageStats, parent: QWidget | None = None):
#         super().__init__(parent)
#         self.setWindowTitle("Гистограмма RGB")
#         self.setModal(True)
#         self.setAttribute(Qt.WA_DeleteOnClose, True)
#         self.resize(900, 550)

#         fig = Figure(figsize=(9, 5), tight_layout=True)
#         canvas = FigureCanvas(fig)
#         ax = fig.add_subplot(111)

#         x = np.arange(256)
#         ax.bar(x - 0.3, stats.hist_r, width=0.3, label="R", color="red", alpha=0.6)
#         ax.bar(x,         stats.hist_g, width=0.3, label="G", color="green", alpha=0.6)
#         ax.bar(x + 0.3,   stats.hist_b, width=0.3, label="B", color="blue", alpha=0.6)

#         ax.set_title("Гистограммы интенсивностей (0..255)")
#         ax.set_xlabel("Интенсивность")
#         ax.set_ylabel("Число пикселей")
#         ax.set_xlim(0, 255)
#         ax.legend()
#         ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

#         buttons = QDialogButtonBox(QDialogButtonBox.Close)
#         buttons.rejected.connect(self.reject)

#         lay = QVBoxLayout(self)
#         lay.addWidget(canvas)
#         lay.addWidget(buttons)


class HistogramDialog(QDialog):
    """Три столбца — ΣR, ΣG, ΣB по изображению."""

    def __init__(self, stats: ImageStats, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Диаграмма каналов (ΣR, ΣG, ΣB)")
        self.setModal(True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(700, 450)

        fig = Figure(figsize=(7, 4), tight_layout=True)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        values = np.array([stats.sum_r, stats.sum_g, stats.sum_b], dtype=np.float64)
        labels = ["R", "G", "B"]
        colors = ["red", "green", "blue"]

        _bars = ax.bar(range(3), values, color=colors, alpha=0.8, width=0.6)
        ax.set_xticks(range(3), labels)
        ax.set_ylabel("Сумма значений канала")
        ax.set_title("Вклад каналов: ΣR, ΣG, ΣB")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

        for i, v in enumerate(values):
            ax.text(i, v, f"{int(v):,}".replace(",", " "), ha="center", va="bottom", fontsize=10)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(canvas)
        lay.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1: Анализ изображений RGB")
        self.resize(1000, 700)

        self.current_pixmap: QPixmap | None = None
        self.current_array: np.ndarray | None = None
        self.current_stats: ImageStats | None = None

        # ---- Виджеты ----
        self.image_label = QLabel("Откройте одно из изображений")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #111; color: #bbb; border-radius: 12px;")

        # Кнопки
        self.btn_img1 = QPushButton("Изображение 1")
        self.btn_img2 = QPushButton("Изображение 2")
        self.btn_hist = QPushButton("Показать гистограмму")
        self.btn_hist.setEnabled(False)

        self.btn_img1.clicked.connect(lambda: self.load_image(IMAGE1_PATH))
        self.btn_img2.clicked.connect(lambda: self.load_image(IMAGE2_PATH))
        self.btn_hist.clicked.connect(self.show_histogram)

        # Блок статистики
        self.stats_group = QGroupBox("Статистика по изображению")
        self.lbl_size = QLabel("Размер: —")
        self.lbl_pixels = QLabel("Пикселей всего: —")
        self.lbl_sum = QLabel("Суммы по каналам Σ: R=—, G=—, B=—")
        self.lbl_share = QLabel("Доля канала в Σ(R+G+B): R=—, G=—, B=—")

        stats_layout = QGridLayout()
        stats_layout.addWidget(self.lbl_size, 0, 0)
        stats_layout.addWidget(self.lbl_pixels, 0, 1)
        stats_layout.addWidget(self.lbl_sum, 1, 0, 1, 2)
        stats_layout.addWidget(self.lbl_share, 2, 0, 1, 2)
        self.stats_group.setLayout(stats_layout)

        # Компоновка
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_img1)
        btn_row.addWidget(self.btn_img2)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_hist)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(btn_row)
        root_layout.addWidget(self.image_label, stretch=1)
        root_layout.addWidget(self.stats_group)
        self.setCentralWidget(root)

    # Вычисления
    def load_image(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(
                self,
                "Файл не найден",
                f"Изображение не найдено:\n{path}\n\n"
                "Проверьте корректность путей IMAGE1_PATH/IMAGE2_PATH.",
            )
            return

        try:
            # Для отображения
            pix = QPixmap(str(path))
            if pix.isNull():
                raise ValueError("Не удалось загрузить QPixmap")

            # Для анализа (PIL → numpy, RGB)
            pil_img = Image.open(path).convert("RGB")
            arr = np.asarray(pil_img, dtype=np.uint8)  # (H, W, 3)

            self.current_pixmap = pix
            self.current_array = arr

            self.update_image_view()
            self.current_stats = self.compute_stats(arr)
            self.update_stats_view(self.current_stats)

            self.btn_hist.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить изображение:\n{e}")

    def update_image_view(self) -> None:
        if self.current_pixmap is None:
            return
        label_size: QSize = self.image_label.size()
        scaled = self.current_pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_image_view()

    @staticmethod
    def compute_stats(arr: np.ndarray) -> ImageStats:
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Ожидался RGB-массив формы (H, W, 3)")

        h, w, _ = arr.shape
        total = int(h * w)

        r = arr[:, :, 0].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 2].astype(np.float64)

        # Количество (сумма значений каналов)
        sum_r = int(r.sum())
        sum_g = int(g.sum())
        sum_b = int(b.sum())

        total_sum = sum_r + sum_g + sum_b
        if total_sum > 0:
            share_r = sum_r / total_sum
            share_g = sum_g / total_sum
            share_b = sum_b / total_sum
        else:
            share_r = share_g = share_b = 0.0

        hist_r, _ = np.histogram(arr[:, :, 0].ravel(), bins=256, range=(0, 256))
        hist_g, _ = np.histogram(arr[:, :, 1].ravel(), bins=256, range=(0, 256))
        hist_b, _ = np.histogram(arr[:, :, 2].ravel(), bins=256, range=(0, 256))

        return ImageStats(
            width=w,
            height=h,
            total_pixels=total,
            sum_r=sum_r,
            sum_g=sum_g,
            sum_b=sum_b,
            share_r=share_r,
            share_g=share_g,
            share_b=share_b,
            hist_r=hist_r.astype(np.int64),
            hist_g=hist_g.astype(np.int64),
            hist_b=hist_b.astype(np.int64),
        )

    def update_stats_view(self, stats: ImageStats) -> None:
        self.lbl_size.setText(f"Размер: {stats.width}×{stats.height} пикс")
        self.lbl_pixels.setText(f"Пикселей всего: {stats.total_pixels}")
        self.lbl_sum.setText(
            f"Σ по каналам: R={stats.sum_r:,}  G={stats.sum_g:,}  B={stats.sum_b:,}".replace(",", " ")
        )
        self.lbl_share.setText(
            "Доля в Σ(R+G+B): "
            f"R={stats.share_r*100:.1f}%  G={stats.share_g*100:.1f}%  B={stats.share_b*100:.1f}%"
        )

    def show_histogram(self) -> None:
        if not self.current_stats:
            return
        dlg = HistogramDialog(self.current_stats, self)
        dlg.exec()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
