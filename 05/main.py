import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class LightSource:
    x: float
    y: float
    z: float
    I0: float
    color: Tuple[float, float, float]


@dataclass
class Sphere:
    cx: float
    cy: float
    cz: float
    radius: float
    color: Tuple[float, float, float]


@dataclass
class Viewer:
    z: float


@dataclass
class ShadingParams:
    k_a: float
    k_d: float
    k_s: float
    shininess: float
    I_ambient: float


@dataclass
class Screen:
    W_mm: float
    H_mm: float
    W_res: int
    H_res: int


class TwoSpheresRenderer:
    def __init__(self, screen: Screen, lights: List[LightSource],
                 spheres: List[Sphere], viewer: Viewer, shading: ShadingParams):
        assert len(spheres) == 2
        self.screen = screen
        self.lights = lights
        self.spheres = spheres
        self.viewer = viewer
        self.shading = shading
        self.rgb_img = None
        self.x_vals = None
        self.y_vals = None
        self.view_axis = "z"
        self.axis_labels = ("X", "Y")
        self.view_title = "Render"

    @staticmethod
    def _normalize(vx, vy, vz, mask=None, eps=1e-10):
        vx = np.asarray(vx, dtype=float)
        vy = np.asarray(vy, dtype=float)
        vz = np.asarray(vz, dtype=float)

        if mask is None:
            mask = np.ones_like(vx, dtype=bool)

        nx = np.zeros_like(vx, dtype=float)
        ny = np.zeros_like(vy, dtype=float)
        nz = np.zeros_like(vz, dtype=float)

        length = np.sqrt(np.maximum(vx * vx + vy * vy + vz * vz, eps))
        valid = mask & (length > eps)

        nx[valid] = vx[valid] / length[valid]
        ny[valid] = vy[valid] / length[valid]
        nz[valid] = vz[valid] / length[valid]
        return nx, ny, nz

    @staticmethod
    def _ray_sphere_intersection(Ox, Oy, Oz, Dx, Dy, Dz, Cx, Cy, Cz, R):
        ocx = Ox - Cx
        ocy = Oy - Cy
        ocz = Oz - Cz

        a = Dx * Dx + Dy * Dy + Dz * Dz
        b = 2.0 * (Dx * ocx + Dy * ocy + Dz * ocz)
        c = ocx * ocx + ocy * ocy + ocz * ocz - R * R

        discriminant = b * b - 4.0 * a * c

        hit = discriminant > 0.0
        t = np.full_like(Dx, np.inf, dtype=float)

        sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a + 1e-10)
        t2 = (-b + sqrt_disc) / (2.0 * a + 1e-10)

        t_candidate = np.where((t1 > 1e-6) & (t1 < t2), t1, t2)
        hit = hit & (t_candidate > 1e-6)
        t[hit] = t_candidate[hit]

        return t, hit

    def compute(self, view_axis="z"):
        view_axis = view_axis.lower()
        if view_axis not in ("z", "x", "y"):
            raise ValueError("view_axis must be one of 'x', 'y', 'z'")
        self.view_axis = view_axis

        Wres = self.screen.W_res
        Hres = self.screen.H_res

        W = self.screen.W_mm
        H = self.screen.H_mm

        def pad_and_match(min_a, max_a, min_b, max_b, pad_frac=0.1):
            if max_a <= min_a:
                max_a = min_a + 1.0
            if max_b <= min_b:
                max_b = min_b + 1.0
            width = max_a - min_a
            height = max_b - min_b
            a0 = min_a - pad_frac * width
            a1 = max_a + pad_frac * width
            b0 = min_b - pad_frac * height
            b1 = max_b + pad_frac * height
            width = a1 - a0
            height = b1 - b0
            target = float(Wres) / float(Hres)
            current = width / height
            if current < target:
                needed = target * height
                extra = needed - width
                a0 -= extra / 2
                a1 += extra / 2
            elif current > target:
                needed = width / target
                extra = needed - height
                b0 -= extra / 2
                b1 += extra / 2
            return a0, a1, b0, b1

        if view_axis == "z":
            xs = np.linspace(-W / 2, W / 2, Wres)
            ys = np.linspace(-H / 2, H / 2, Hres)
            Xs, Ys = np.meshgrid(xs, ys)
            self.axis_labels = ("X", "Y")
            self.view_title = "Front view (-Z)"
        elif view_axis == "x":
            y_min = min(s.cy - s.radius for s in self.spheres)
            y_max = max(s.cy + s.radius for s in self.spheres)
            z_min = min(s.cz - s.radius for s in self.spheres)
            z_max = max(s.cz + s.radius for s in self.spheres)
            y0, y1, z0, z1 = pad_and_match(y_min, y_max, z_min, z_max)
            xs = np.linspace(y0, y1, Wres)  # image X = world Y
            ys = np.linspace(z0, z1, Hres)  # image Y = world Z
            Xs, Ys = np.meshgrid(xs, ys)
            self.axis_labels = ("Y", "Z")
            self.view_title = "Side view (-X)"
        else:
            x_min = min(s.cx - s.radius for s in self.spheres)
            x_max = max(s.cx + s.radius for s in self.spheres)
            z_min = min(s.cz - s.radius for s in self.spheres)
            z_max = max(s.cz + s.radius for s in self.spheres)
            x0, x1, z0, z1 = pad_and_match(x_min, x_max, z_min, z_max)
            xs = np.linspace(x0, x1, Wres)  # image X = world X
            ys = np.linspace(z0, z1, Hres)  # image Y = world Z
            Xs, Ys = np.meshgrid(xs, ys)
            self.axis_labels = ("X", "Z")
            self.view_title = "Top view (-Y)"

        self.x_vals = xs
        self.y_vals = ys

        dist = self.viewer.z
        if view_axis == "z":
            Ox, Oy, Oz = 0.0, 0.0, dist
            plane_x, plane_y, plane_z = Xs, Ys, np.zeros_like(Xs)
            Dx = plane_x - Ox
            Dy = plane_y - Oy
            Dz = plane_z - Oz
            Dx, Dy, Dz = self._normalize(Dx, Dy, Dz)
        elif view_axis == "x":
            Ox = np.full_like(Xs, dist)
            Oy = Xs
            Oz = Ys
            Dx = np.ones_like(Xs)
            Dy = np.zeros_like(Xs)
            Dz = np.zeros_like(Xs)
            Dx, Dy, Dz = self._normalize(Dx, Dy, Dz)
        else:
            Ox = Xs
            Oy = np.full_like(Xs, dist)
            Oz = Ys
            Dx = np.zeros_like(Xs)
            Dy = np.ones_like(Xs)
            Dz = np.zeros_like(Xs)
            Dx, Dy, Dz = self._normalize(Dx, Dy, Dz)

        t_hit = np.full_like(Xs, np.inf, dtype=float)
        sphere_id = np.full_like(Xs, -1, dtype=int)
        hit_mask = np.zeros_like(Xs, dtype=bool)

        for idx, sphere in enumerate(self.spheres):
            t, hit = self._ray_sphere_intersection(
                Ox, Oy, Oz, Dx, Dy, Dz,
                sphere.cx, sphere.cy, sphere.cz, sphere.radius
            )
            mask = hit & (t < t_hit)
            t_hit[mask] = t[mask]
            sphere_id[mask] = idx
            hit_mask = hit_mask | mask

        if not np.any(hit_mask):
            self.rgb_img = np.zeros((Hres, Wres, 3), dtype=np.uint8)
            return self.rgb_img

        Px = Ox + Dx * t_hit
        Py = Oy + Dy * t_hit
        Pz = Oz + Dz * t_hit

        Nx = np.zeros_like(Px)
        Ny = np.zeros_like(Py)
        Nz = np.zeros_like(Pz)

        for idx, sphere in enumerate(self.spheres):
            mask = (sphere_id == idx) & hit_mask
            if np.any(mask):
                Nx[mask] = (Px[mask] - sphere.cx) / sphere.radius
                Ny[mask] = (Py[mask] - sphere.cy) / sphere.radius
                Nz[mask] = (Pz[mask] - sphere.cz) / sphere.radius

        Nx, Ny, Nz = self._normalize(Nx, Ny, Nz, mask=hit_mask)

        Vx = Ox - Px
        Vy = Oy - Py
        Vz = Oz - Pz
        Vx, Vy, Vz = self._normalize(Vx, Vy, Vz, mask=hit_mask)

        R_img = np.zeros_like(Px, dtype=float)
        G_img = np.zeros_like(Px, dtype=float)
        B_img = np.zeros_like(Px, dtype=float)

        Ia = float(self.shading.I_ambient)
        ka = float(self.shading.k_a)
        kd = float(self.shading.k_d)
        ks = float(self.shading.k_s)
        n = float(self.shading.shininess)

        for idx, sphere in enumerate(self.spheres):
            mask = sphere_id == idx
            if np.any(mask):
                sr, sg, sb = sphere.color
                R_img[mask] += ka * Ia * sr
                G_img[mask] += ka * Ia * sg
                B_img[mask] += ka * Ia * sb

        for light in self.lights:
            Lx, Ly, Lz = light.x, light.y, light.z
            Lr, Lg, Lb = light.color
            I0 = light.I0

            Lvx = Lx - Px
            Lvy = Ly - Py
            Lvz = Lz - Pz

            dist_sq = Lvx * Lvx + Lvy * Lvy + Lvz * Lvz
            dist = np.sqrt(dist_sq)

            Lvxn, Lvyn, Lvzn = self._normalize(Lvx, Lvy, Lvz, mask=hit_mask)

            attenuation = np.zeros_like(dist)
            valid_dist = dist > 1e-6
            attenuation[valid_dist] = I0 / (dist_sq[valid_dist] + 1e-10)

            ndotl = Nx * Lvxn + Ny * Lvyn + Nz * Lvzn
            ndotl = np.maximum(ndotl, 0.0)

            shadow = np.zeros_like(Px, dtype=bool)

            for sphere_idx, sphere in enumerate(self.spheres):
                if sphere_idx == 0:
                    point_mask = (sphere_id == 0) & hit_mask
                    other_sphere = self.spheres[1]
                else:
                    point_mask = (sphere_id == 1) & hit_mask
                    other_sphere = self.spheres[0]

                if not np.any(point_mask):
                    continue

                ray_origin_x = Px[point_mask]
                ray_origin_y = Py[point_mask]
                ray_origin_z = Pz[point_mask]

                ray_dir_x = Lvxn[point_mask]
                ray_dir_y = Lvyn[point_mask]
                ray_dir_z = Lvzn[point_mask]

                t_shadow, hit_shadow = self._ray_sphere_intersection(
                    ray_origin_x, ray_origin_y, ray_origin_z,
                    ray_dir_x, ray_dir_y, ray_dir_z,
                    other_sphere.cx, other_sphere.cy, other_sphere.cz,
                    other_sphere.radius
                )

                dist_to_light = dist[point_mask]
                shadow_mask = hit_shadow & (t_shadow > 1e-4) & (t_shadow < dist_to_light - 1e-4)

                shadow_indices = np.where(point_mask)
                temp_shadow = np.zeros_like(Px, dtype=bool)

                for i in range(len(shadow_indices[0])):
                    idx_y, idx_x = shadow_indices[0][i], shadow_indices[1][i]
                    if i < len(shadow_mask):
                        temp_shadow[idx_y, idx_x] = shadow_mask.flat[i]

                shadow = shadow | temp_shadow

            diffuse = np.zeros_like(Px)
            light_mask = hit_mask & (~shadow) & (ndotl > 0)
            diffuse[light_mask] = kd * ndotl[light_mask] * attenuation[light_mask]

            specular = np.zeros_like(Px)

            if ks > 0:
                Hx = Lvxn + Vx
                Hy = Lvyn + Vy
                Hz = Lvzn + Vz
                Hx, Hy, Hz = self._normalize(Hx, Hy, Hz, mask=light_mask)

                ndoth = Nx * Hx + Ny * Hy + Nz * Hz
                ndoth = np.maximum(ndoth, 0.0)

                specular_mask = light_mask & (ndoth > 0)
                specular[specular_mask] = ks * (ndoth[specular_mask] ** n) * attenuation[specular_mask]

            for idx, sphere in enumerate(self.spheres):
                mask = (sphere_id == idx) & light_mask
                if np.any(mask):
                    sr, sg, sb = sphere.color

                    R_img[mask] += sr * diffuse[mask] * Lr
                    G_img[mask] += sg * diffuse[mask] * Lg
                    B_img[mask] += sb * diffuse[mask] * Lb

                    R_img[mask] += specular[mask] * Lr
                    G_img[mask] += specular[mask] * Lg
                    B_img[mask] += specular[mask] * Lb

        max_val = max(np.max(R_img), np.max(G_img), np.max(B_img), 1.0)

        R_norm = np.clip(R_img / max_val, 0.0, 1.0)
        G_norm = np.clip(G_img / max_val, 0.0, 1.0)
        B_norm = np.clip(B_img / max_val, 0.0, 1.0)

        gamma = 1.0 / 2.2
        R_gamma = R_norm ** gamma
        G_gamma = G_norm ** gamma
        B_gamma = B_norm ** gamma

        img = np.stack([R_gamma, G_gamma, B_gamma], axis=-1)
        img = (img * 255).astype(np.uint8)

        img[~hit_mask] = [0, 0, 0]

        self.rgb_img = img
        return img

    def save_image(self, filename: str):
        if self.rgb_img is None:
            return
        Image.fromarray(self.rgb_img, mode="RGB").save(filename)

    def show_image(self):
        if self.rgb_img is None:
            return

        plt.figure(figsize=(10, 8))
        plt.imshow(
            self.rgb_img,
            extent=[self.x_vals[0], self.x_vals[-1],
                    self.y_vals[0], self.y_vals[-1]],
            origin='lower'
        )
        title = getattr(self, "view_title", "Render")
        plt.title(title)
        x_label, y_label = getattr(self, "axis_labels", ("X", "Y"))
        plt.xlabel(f"{x_label}, mm")
        plt.ylabel(f"{y_label}, mm")
        plt.tight_layout()
        plt.show()

    def show_scene_projections(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        colors = ['blue', 'orange']
        for idx, sphere in enumerate(self.spheres):
            circle = plt.Circle((sphere.cx, sphere.cy), sphere.radius,
                               fill=False, color=colors[idx], linewidth=2)
            ax.add_patch(circle)
            ax.plot(sphere.cx, sphere.cy, 'o', color=colors[idx],
                   label=f'Сфера {idx + 1}')

        for i, light in enumerate(self.lights):
            ax.plot(light.x, light.y, '*', markersize=12,
                   label=f'Источник {i + 1}')

        ax.plot(0, 0, '^', markersize=10, label='Наблюдатель')
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.set_title('Проекция XY')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        ax = axes[1]
        for idx, sphere in enumerate(self.spheres):
            circle = plt.Circle((sphere.cx, sphere.cz), sphere.radius,
                               fill=False, color=colors[idx], linewidth=2)
            ax.add_patch(circle)
            ax.plot(sphere.cx, sphere.cz, 'o', color=colors[idx])

        for i, light in enumerate(self.lights):
            ax.plot(light.x, light.z, '*', markersize=12)

        ax.plot(0, self.viewer.z, '^', markersize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Экран (z=0)')
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Z, мм')
        ax.set_title('Проекция XZ')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        ax = axes[2]
        for idx, sphere in enumerate(self.spheres):
            circle = plt.Circle((sphere.cy, sphere.cz), sphere.radius,
                               fill=False, color=colors[idx], linewidth=2)
            ax.add_patch(circle)
            ax.plot(sphere.cy, sphere.cz, 'o', color=colors[idx])

        for i, light in enumerate(self.lights):
            ax.plot(light.y, light.z, '*', markersize=12)

        ax.plot(0, self.viewer.z, '^', markersize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Y, мм')
        ax.set_ylabel('Z, мм')
        ax.set_title('Проекция YZ')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        plt.tight_layout()
        plt.show()


class TwoSpheresApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа №5: Тени и освещение")

        self.geometry("500x700")

        self.canvas = tk.Canvas(self, borderwidth=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.content = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self.entries = {}
        self.light_entries = []
        self.light_row_widgets = []
        self.light_frame = None
        self.output_path = tk.StringVar(value="result.png")
        self.status_var = tk.StringVar(value="Готово к работе")

        self._build_ui()

    def _add_field(self, parent, row, label, default):
        tk.Label(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(parent, textvariable=var, width=15)
        ent.grid(row=row, column=1, padx=5, pady=2)
        self.entries[label] = var

    def _add_light_row(self, defaults=None):
        if defaults is None:
            defaults = {
                "x": -500.0, "y": 500.0, "z": 1500.0,
                "I0": 1000.0, "R": 1.0, "G": 1.0, "B": 1.0
            }

        row_index = len(self.light_entries) + 1

        if row_index == 1:
            headers = ["X", "Y", "Z", "I0", "R", "G", "B"]
            for col, header in enumerate(headers):
                tk.Label(self.light_frame, text=header, width=8,
                        relief="ridge").grid(row=0, column=col, padx=2, pady=2)

        vars_row = {}
        row_widgets = []

        for col, key in enumerate(["x", "y", "z", "I0", "R", "G", "B"]):
            value = defaults.get(key, 0.0)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(self.light_frame, textvariable=var, width=8)
            entry.grid(row=row_index, column=col, padx=2, pady=2)
            row_widgets.append(entry)
            vars_row[key] = var

        self.light_entries.append(vars_row)
        self.light_row_widgets.append(row_widgets)

    def _remove_light_row(self):
        if len(self.light_entries) <= 1:
            messagebox.showwarning("Внимание", "Должен остаться хотя бы один источник света")
            return

        self.light_entries.pop()
        row_widgets = self.light_row_widgets.pop()
        for widget in row_widgets:
            widget.destroy()

    def _build_ui(self):
        row = 0

        ttk.Label(self.content, text="Параметры экрана", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2)
        )
        row += 1

        self._add_field(self.content, row, "Ширина (мм)", 800)
        row += 1
        self._add_field(self.content, row, "Высота (мм)", 600)
        row += 1
        self._add_field(self.content, row, "Разрешение X", 400)
        row += 1
        self._add_field(self.content, row, "Разрешение Y", 300)
        row += 1

        ttk.Label(self.content, text="Источники света", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2)
        )
        row += 1

        self.light_frame = ttk.LabelFrame(self.content, text="Параметры источников")
        self.light_frame.grid(row=row, column=0, columnspan=3,
                             sticky="we", padx=5, pady=5)
        row += 1

        self._add_light_row()

        btn_frame = ttk.Frame(self.content)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=5)
        ttk.Button(btn_frame, text="+ Добавить источник",
                  command=self._add_light_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="- Удалить источник",
                  command=self._remove_light_row).pack(side=tk.LEFT, padx=5)
        row += 1

        ttk.Label(self.content, text="Сфера 1", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2)
        )
        row += 1

        self._add_field(self.content, row, "Центр X1", -150)
        row += 1
        self._add_field(self.content, row, "Центр Y1", 0)
        row += 1
        self._add_field(self.content, row, "Центр Z1", 1200)
        row += 1
        self._add_field(self.content, row, "Радиус R1", 300)
        row += 1
        self._add_field(self.content, row, "Цвет R1", 0.8)
        row += 1
        self._add_field(self.content, row, "Цвет G1", 0.3)
        row += 1
        self._add_field(self.content, row, "Цвет B1", 0.3)
        row += 1

        ttk.Label(self.content, text="Сфера 2", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2)
        )
        row += 1

        self._add_field(self.content, row, "Центр X2", 200)
        row += 1
        self._add_field(self.content, row, "Центр Y2", 100)
        row += 1
        self._add_field(self.content, row, "Центр Z2", 1000)
        row += 1
        self._add_field(self.content, row, "Радиус R2", 180)
        row += 1
        self._add_field(self.content, row, "Цвет R2", 0.3)
        row += 1
        self._add_field(self.content, row, "Цвет G2", 0.3)
        row += 1
        self._add_field(self.content, row, "Цвет B2", 0.8)
        row += 1

        ttk.Label(self.content, text="Параметры рендеринга", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2)
        )
        row += 1

        self._add_field(self.content, row, "Позиция наблюдателя Z", -800)
        row += 1
        self._add_field(self.content, row, "Коэф. ambient", 0.2)
        row += 1
        self._add_field(self.content, row, "Коэф. diffuse", 0.7)
        row += 1
        self._add_field(self.content, row, "Коэф. specular", 0.5)
        row += 1
        self._add_field(self.content, row, "Блеск", 32)
        row += 1
        self._add_field(self.content, row, "Фоновое освещение", 0.1)
        row += 1

        ttk.Label(self.content, text="Выходной файл").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(self.content, textvariable=self.output_path, width=25).grid(
            row=row, column=1, padx=5, pady=2
        )
        ttk.Button(self.content, text="...", width=3,
                  command=self._browse_file).grid(
            row=row, column=2, padx=5, pady=2
        )
        row += 1

        status_label = tk.Label(self.content, textvariable=self.status_var,
                               anchor="w", fg="blue")
        status_label.grid(row=row, column=0, columnspan=3,
                         sticky="we", padx=5, pady=2)
        row += 1

        ttk.Button(self.content, text="Рассчитать и визуализировать",
                  command=self._on_calculate).grid(
            row=row, column=0, columnspan=3, pady=15, padx=20
        )

        row += 1
        ttk.Button(self.content, text="Render 3 views (Z/X/Y)",
                  command=self._on_calculate_views).grid(
            row=row, column=0, columnspan=3, pady=5, padx=20
        )

    def _browse_file(self):
        fname = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if fname:
            self.output_path.set(fname)

    def _read_parameters(self):
        try:
            W_mm = float(self.entries["Ширина (мм)"].get())
            H_mm = float(self.entries["Высота (мм)"].get())
            W_res = int(self.entries["Разрешение X"].get())
            H_res = int(self.entries["Разрешение Y"].get())

            if W_mm <= 0 or H_mm <= 0:
                raise ValueError("Размеры экрана должны быть положительными")
            if W_res < 100 or H_res < 100:
                raise ValueError("Разрешение должно быть не менее 100 пикселей")

            lights = []
            for idx, vars_row in enumerate(self.light_entries):
                x = float(vars_row["x"].get())
                y = float(vars_row["y"].get())
                z = float(vars_row["z"].get())
                I0 = float(vars_row["I0"].get())
                R = float(vars_row["R"].get())
                G = float(vars_row["G"].get())
                B = float(vars_row["B"].get())

                if I0 <= 0:
                    raise ValueError(f"Интенсивность источника {idx + 1} должна быть положительной")

                lights.append(LightSource(x=x, y=y, z=z, I0=I0, color=(R, G, B)))

            if not lights:
                raise ValueError("Должен быть хотя бы один источник света")

            x1 = float(self.entries["Центр X1"].get())
            y1 = float(self.entries["Центр Y1"].get())
            z1 = float(self.entries["Центр Z1"].get())
            r1 = float(self.entries["Радиус R1"].get())
            cr1 = float(self.entries["Цвет R1"].get())
            cg1 = float(self.entries["Цвет G1"].get())
            cb1 = float(self.entries["Цвет B1"].get())

            x2 = float(self.entries["Центр X2"].get())
            y2 = float(self.entries["Центр Y2"].get())
            z2 = float(self.entries["Центр Z2"].get())
            r2 = float(self.entries["Радиус R2"].get())
            cr2 = float(self.entries["Цвет R2"].get())
            cg2 = float(self.entries["Цвет G2"].get())
            cb2 = float(self.entries["Цвет B2"].get())

            if r1 <= 0 or r2 <= 0:
                raise ValueError("Радиусы сфер должны быть положительными")

            for name, val in [("R1", cr1), ("G1", cg1), ("B1", cb1),
                             ("R2", cr2), ("G2", cg2), ("B2", cb2)]:
                if val < 0 or val > 1:
                    raise ValueError(f"Цвет {name} должен быть в диапазоне [0, 1]")

            z_viewer = float(self.entries["Позиция наблюдателя Z"].get())
            if z_viewer >= 0:
                raise ValueError("Наблюдатель должен находиться перед экраном (Z < 0)")

            k_a = float(self.entries["Коэф. ambient"].get())
            k_d = float(self.entries["Коэф. diffuse"].get())
            k_s = float(self.entries["Коэф. specular"].get())
            shininess = float(self.entries["Блеск"].get())
            I_amb = float(self.entries["Фоновое освещение"].get())

            if k_a < 0 or k_d < 0 or k_s < 0 or I_amb < 0:
                raise ValueError("Коэффициенты освещения должны быть неотрицательными")
            if shininess <= 0:
                raise ValueError("Коэффициент блеска должен быть положительным")

            screen = Screen(W_mm=W_mm, H_mm=H_mm, W_res=W_res, H_res=H_res)
            sphere1 = Sphere(cx=x1, cy=y1, cz=z1, radius=r1, color=(cr1, cg1, cb1))
            sphere2 = Sphere(cx=x2, cy=y2, cz=z2, radius=r2, color=(cr2, cg2, cb2))
            viewer = Viewer(z=z_viewer)
            shading = ShadingParams(k_a=k_a, k_d=k_d, k_s=k_s,
                                   shininess=shininess, I_ambient=I_amb)

            return screen, lights, [sphere1, sphere2], viewer, shading

        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return None

    def _on_calculate(self):
        params = self._read_parameters()
        if params is None:
            return

        screen, lights, spheres, viewer, shading = params

        self.status_var.set("Вычисление...")
        self.update()

        try:
            renderer = TwoSpheresRenderer(screen, lights, spheres, viewer, shading)
            renderer.compute("z")

            renderer.save_image(self.output_path.get())
            renderer.show_image()
            renderer.show_scene_projections()

            self.status_var.set(f"Готово! Сохранено в {self.output_path.get()}")

        except Exception as e:
            messagebox.showerror("Ошибка расчета", str(e))
            self.status_var.set("Ошибка при расчете")

    def _on_calculate_views(self):
        params = self._read_parameters()
        if params is None:
            return

        screen, lights, spheres, viewer, shading = params
        base = Path(self.output_path.get())
        views = [("z", "front"), ("x", "side"), ("y", "top")]
        saved = []

        try:
            for axis, suffix in views:
                renderer = TwoSpheresRenderer(screen, lights, spheres, viewer, shading)
                renderer.compute(axis)
                out_path = base.parent / f"{base.stem}_{suffix}{base.suffix}"
                renderer.save_image(str(out_path))
                renderer.show_image()
                saved.append(str(out_path))

            messagebox.showinfo("Готово", "Сохранены картинки:\n" + "\n".join(saved))
        except Exception as e:
            messagebox.showerror("Ошибка отображения", str(e))


def main():
    app = TwoSpheresApp()
    app.mainloop()


if __name__ == "__main__":
    main()
