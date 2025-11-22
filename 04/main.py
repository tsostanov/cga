import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class LightSource:
    x: float
    y: float
    z: float
    I0: float


@dataclass
class Sphere:
    cx: float
    cy: float
    cz: float
    radius: float


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


class SphereRenderer:
    def __init__(self, screen: Screen, lights: List[LightSource],
                 sphere: Sphere, viewer: Viewer, shading: ShadingParams):
        self.screen = screen
        self.lights = lights
        self.sphere = sphere
        self.viewer = viewer
        self.shading = shading

        self.I_img = None
        self.I_raw = None
        self.x_vals = None
        self.y_vals = None
        self.mask_sphere = None

    def compute(self):
        W_mm = self.screen.W_mm
        H_mm = self.screen.H_mm
        W_res = self.screen.W_res
        H_res = self.screen.H_res

        x_min, x_max = -W_mm / 2, W_mm / 2
        y_min, y_max = -H_mm / 2, H_mm / 2

        x_vals = np.linspace(x_min, x_max, W_res)
        y_vals = np.linspace(y_min, y_max, H_res)
        X, Y = np.meshgrid(x_vals, y_vals)

        self.x_vals = x_vals
        self.y_vals = y_vals

        dx = X - self.sphere.cx
        dy = Y - self.sphere.cy
        rho2 = dx ** 2 + dy ** 2
        R2 = self.sphere.radius ** 2

        mask_sphere = rho2 <= R2
        self.mask_sphere = mask_sphere

        z_offset = np.zeros_like(X)
        z_offset[mask_sphere] = np.sqrt(R2 - rho2[mask_sphere])

        if self.viewer.z < self.sphere.cz:
            Z = self.sphere.cz - z_offset
        else:
            Z = self.sphere.cz + z_offset

        Px = X
        Py = Y
        Pz = Z

        Nx = (Px - self.sphere.cx) / self.sphere.radius
        Ny = (Py - self.sphere.cy) / self.sphere.radius
        Nz = (Pz - self.sphere.cz) / self.sphere.radius

        k_a = self.shading.k_a
        k_d = self.shading.k_d
        k_s = self.shading.k_s
        n = self.shading.shininess
        I_amb = self.shading.I_ambient

        Vx = -Px
        Vy = -Py
        Vz = self.viewer.z - Pz
        r_V = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2)
        r_V[r_V == 0] = 1e-6
        Vx /= r_V
        Vy /= r_V
        Vz /= r_V

        I_diffuse = np.zeros_like(X, dtype=float)
        I_specular = np.zeros_like(X, dtype=float)

        for light in self.lights:
            Lx = light.x - Px
            Ly = light.y - Py
            Lz = light.z - Pz
            r_L = np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)
            r_L[r_L == 0] = 1e-6
            Lx /= r_L
            Ly /= r_L
            Lz /= r_L

            Hx = Lx + Vx
            Hy = Ly + Vy
            Hz = Lz + Vz
            r_H = np.sqrt(Hx ** 2 + Hy ** 2 + Hz ** 2)
            r_H[r_H == 0] = 1e-6
            Hx /= r_H
            Hy /= r_H
            Hz /= r_H

            N_dot_L = np.clip(Nx * Lx + Ny * Ly + Nz * Lz, 0.0, None)
            N_dot_H = np.clip(Nx * Hx + Ny * Hy + Nz * Hz, 0.0, None)
            attenuation = light.I0 / (r_L ** 2)

            I_diffuse += k_d * attenuation * N_dot_L
            I_specular += k_s * attenuation * (N_dot_H ** n)

        I = np.zeros_like(X, dtype=float)
        I[mask_sphere] = k_a * I_amb + I_diffuse[mask_sphere] + I_specular[mask_sphere]
        self.I_raw = I

        I_inside = I[mask_sphere]
        if I_inside.size == 0:
            I_min, I_max = 0.0, 1.0
        else:
            I_min = np.min(I_inside)
            I_max = np.max(I_inside)
            if I_max - I_min < 1e-9:
                I_min = 0.0
                if I_max == 0:
                    I_max = 1.0

        I_norm = np.zeros_like(I, dtype=float)
        if I_max > I_min:
            I_norm[mask_sphere] = (I[mask_sphere] - I_min) / (I_max - I_min) * 255.0

        I_img = np.clip(I_norm, 0, 255).astype(np.uint8)
        self.I_img = I_img

        return I_img

    def brightness_at_xy(self, x_2d: float, y_2d: float):
        dx = x_2d - self.sphere.cx
        dy = y_2d - self.sphere.cy
        rho2 = dx * dx + dy * dy
        R2 = self.sphere.radius ** 2

        if rho2 > R2:
            return None

        z_offset = np.sqrt(R2 - rho2)

        if self.viewer.z < self.sphere.cz:
            z_surf = self.sphere.cz - z_offset
        else:
            z_surf = self.sphere.cz + z_offset

        P = np.array([x_2d, y_2d, z_surf], dtype=float)
        C = np.array([self.sphere.cx, self.sphere.cy, self.sphere.cz], dtype=float)
        N = (P - C) / self.sphere.radius

        V_vec = np.array([0.0, 0.0, self.viewer.z], dtype=float) - P
        r_V = np.linalg.norm(V_vec)
        if r_V == 0:
            r_V = 1e-6
        V = V_vec / r_V

        k_a = self.shading.k_a
        k_d = self.shading.k_d
        k_s = self.shading.k_s
        n = self.shading.shininess
        I_amb = self.shading.I_ambient

        I_val = k_a * I_amb

        for light in self.lights:
            L_vec = np.array([light.x, light.y, light.z], dtype=float) - P
            r_L = np.linalg.norm(L_vec)
            if r_L == 0:
                r_L = 1e-6
            L = L_vec / r_L

            H_vec = L + V
            r_H = np.linalg.norm(H_vec)
            if r_H == 0:
                H = V
            else:
                H = H_vec / r_H

            N_dot_L = max(0.0, float(np.dot(N, L)))
            N_dot_H = max(0.0, float(np.dot(N, H)))
            attenuation = light.I0 / (r_L ** 2)

            I_diffuse = k_d * attenuation * N_dot_L
            I_specular = k_s * attenuation * (N_dot_H ** n)
            I_val += I_diffuse + I_specular

        return I_val

    def print_control_points(self):
        print("\n--- Контрольные точки на сфере ---")
        R = self.sphere.radius
        cx = self.sphere.cx
        cy = self.sphere.cy

        points_2d = {
            "center": (cx, cy),
            "+X": (cx + R, cy),
            "-X": (cx - R, cy),
            "+Y": (cx, cy + R),
            "-Y": (cx, cy - R),
        }

        for name, (px, py) in points_2d.items():
            val = self.brightness_at_xy(px, py)
            if val is None:
                print(f"{name:6s}: x={px:8.2f} y={py:8.2f} -> вне сферы")
            else:
                print(f"{name:6s}: x={px:8.2f} y={py:8.2f} -> I={val:.6f}")

        if self.I_raw is not None and self.mask_sphere is not None:
            I_inside = self.I_raw[self.mask_sphere]
            if I_inside.size > 0:
                print("\n--- Характеристики яркости по сфере ---")
                print(f"I_max = {np.max(I_inside):.6f}")
                print(f"I_min = {np.min(I_inside):.6f}")
                print(f"I_avg = {np.mean(I_inside):.6f}")
            else:
                print("\n(Сфера не попала в область расчёта)")

    def save_image(self, filename: str):
        if self.I_img is None:
            return
        Image.fromarray(self.I_img, mode="L").save(filename)

    def show_image(self):
        if self.I_img is None:
            return
        x_min, x_max = self.x_vals[0], self.x_vals[-1]
        y_min, y_max = self.y_vals[0], self.y_vals[-1]

        plt.figure()
        plt.imshow(self.I_img, cmap="gray", extent=[x_min, x_max, y_max, y_min])
        circle = plt.Circle(
            (self.sphere.cx, self.sphere.cy),
            self.sphere.radius,
            color="red",
            fill=False,
            linewidth=1,
        )
        plt.gca().add_patch(circle)
        plt.title("Распределение яркости на сфере (проекция)")
        plt.xlabel("x, мм")
        plt.ylabel("y, мм")
        plt.tight_layout()
        plt.show()

    def show_horizontal_section(self):
        if self.I_raw is None:
            return
        center_y = self.sphere.cy
        row_idx = (np.abs(self.y_vals - center_y)).argmin()
        section_x = self.x_vals
        section_I = (self.I_raw * self.mask_sphere)[row_idx, :]

        plt.figure()
        plt.plot(section_x, section_I)
        plt.title("Горизонтальное сечение через центр сферы")
        plt.xlabel("x, мм")
        plt.ylabel("Яркость")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_vertical_section(self):
        if self.I_raw is None:
            return
        center_x = self.sphere.cx
        col_idx = (np.abs(self.x_vals - center_x)).argmin()
        section_y = self.y_vals
        section_I = (self.I_raw * self.mask_sphere)[:, col_idx]

        plt.figure()
        plt.plot(section_y, section_I)
        plt.title("Вертикальное сечение через центр сферы")
        plt.xlabel("y, мм")
        plt.ylabel("Яркость")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class SphereApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Яркость на сфере (Блинн-Фонг)")

        self.entries = {}
        self.light_entries = []
        self.light_row_widgets = []
        self.light_frame = None
        self.output_path = tk.StringVar(value="sphere_brightness.png")
        self.status_var = tk.StringVar(value="")

        self._build_ui()

    def _add_field(self, row, label, default):
        tk.Label(self, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(self, textvariable=var, width=15)
        ent.grid(row=row, column=1, padx=5, pady=2)
        self.entries[label] = var

    def _add_light_row(self, defaults=None):
        if defaults is None:
            defaults = {"x": 0.0, "y": 0.0, "z": 500.0, "I0": 1000.0}
        if self.light_frame is None:
            return
        row_index = len(self.light_entries) + 1
        vars_row = {}
        row_widgets = []
        for col, key in enumerate(("x", "y", "z", "I0")):
            value = defaults.get(key, 0.0)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(self.light_frame, textvariable=var, width=12)
            entry.grid(row=row_index, column=col, padx=3, pady=2)
            row_widgets.append(entry)
            vars_row[key] = var
        self.light_entries.append(vars_row)
        self.light_row_widgets.append(row_widgets)

    def _remove_light_row(self):
        if not self.light_entries:
            return
        if len(self.light_entries) == 1:
            messagebox.showwarning("Предупреждение", "Нужен минимум один источник света.")
            return
        self.light_entries.pop()
        row_widgets = self.light_row_widgets.pop()
        for widget in row_widgets:
            widget.destroy()

    def _build_ui(self):
        row = 0

        self._add_field(row, "W_mm", 1000)
        row += 1
        self._add_field(row, "H_mm", 1000)
        row += 1
        self._add_field(row, "W_res", 400)
        row += 1
        self._add_field(row, "H_res", 400)
        row += 1

        self.light_entries.clear()
        self.light_row_widgets.clear()
        self.light_frame = ttk.LabelFrame(self, text="Источники света")
        self.light_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=5, pady=5)
        for col, text in enumerate(("x_L (мм)", "y_L (мм)", "z_L (мм)", "I0")):
            tk.Label(self.light_frame, text=text, anchor="w").grid(
                row=0, column=col, padx=3, pady=2
            )
        self._add_light_row()
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=row + 1, column=0, columnspan=3, pady=5)
        ttk.Button(controls_frame, text="Добавить источник", command=self._add_light_row).grid(
            row=0, column=0, padx=5
        )
        ttk.Button(controls_frame, text="Удалить источник", command=self._remove_light_row).grid(
            row=0, column=1, padx=5
        )
        row += 2

        self._add_field(row, "x_C", 0.0)
        row += 1
        self._add_field(row, "y_C", 0.0)
        row += 1
        self._add_field(row, "z_C", 2000.0)
        row += 1
        self._add_field(row, "R_sphere", 800.0)
        row += 1

        self._add_field(row, "z_O", 0.0)
        row += 1

        self._add_field(row, "k_a", 0.1)
        row += 1
        self._add_field(row, "k_d", 1.0)
        row += 1
        self._add_field(row, "k_s", 0.8)
        row += 1
        self._add_field(row, "shininess", 50.0)
        row += 1
        self._add_field(row, "I_ambient", 1.0)
        row += 1

        tk.Label(self, text="Файл").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(self, textvariable=self.output_path, width=25).grid(
            row=row, column=1, padx=5, pady=2
        )

        ttk.Button(self, text="...", command=self._browse_file).grid(
            row=row, column=2, padx=5, pady=2
        )
        row += 1

        status_label = tk.Label(self, textvariable=self.status_var, anchor="w", fg="gray")
        status_label.grid(row=row, column=0, columnspan=3, sticky="we", padx=5, pady=2)
        row += 1

        ttk.Button(self, text="Рассчитать и визуализировать",
                   command=self._on_calculate).grid(
            row=row, column=0, columnspan=3, pady=10
        )

    def _browse_file(self):
        fname = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")]
        )
        if fname:
            self.output_path.set(fname)

    def _read_parameters(self):
        def ensure_range(value, min_value, max_value, name):
            if value < min_value or value > max_value:
                raise ValueError(f"{name} должно быть в диапазоне [{min_value}, {max_value}].")

        try:
            W_mm = float(self.entries["W_mm"].get())
            H_mm = float(self.entries["H_mm"].get())
            H_res = int(self.entries["H_res"].get())

            ensure_range(W_mm, 100.0, 10000.0, "W_mm")
            ensure_range(H_mm, 100.0, 10000.0, "H_mm")
            ensure_range(H_res, 200, 800, "H_res")

            W_res = int(round(H_res * W_mm / H_mm))
            ensure_range(W_res, 200, 800, "W_res")
            self.entries["W_res"].set(str(W_res))
            self.status_var.set(
                f"Ширина пересчитана до {W_res} для квадратных пикселей."
            )

            x_C = float(self.entries["x_C"].get())
            y_C = float(self.entries["y_C"].get())
            z_C = float(self.entries["z_C"].get())
            R_sphere = float(self.entries["R_sphere"].get())

            ensure_range(x_C, -10000.0, 10000.0, "x_C")
            ensure_range(y_C, -10000.0, 10000.0, "y_C")
            ensure_range(z_C, 100.0, 10000.0, "z_C")
            if R_sphere <= 0:
                raise ValueError("R_sphere должно быть больше нуля.")

            z_O = float(self.entries["z_O"].get())
            ensure_range(z_O, 0.0, 10000.0, "z_O")

            k_a = float(self.entries["k_a"].get())
            k_d = float(self.entries["k_d"].get())
            k_s = float(self.entries["k_s"].get())
            shininess = float(self.entries["shininess"].get())
            I_amb = float(self.entries["I_ambient"].get())

            for name, value in (("k_a", k_a), ("k_d", k_d), ("k_s", k_s), ("I_ambient", I_amb)):
                if value < 0:
                    raise ValueError(f"{name} должно быть неотрицательным.")
            if shininess <= 0:
                raise ValueError("shininess должно быть больше нуля.")

            lights = []
            for idx, vars_row in enumerate(self.light_entries, start=1):
                values = {key: vars_row[key].get().strip() for key in ("x", "y", "z", "I0")}
                if not any(values.values()):
                    continue
                x_L = float(values["x"])
                y_L = float(values["y"])
                z_L = float(values["z"])
                I0 = float(values["I0"])
                ensure_range(x_L, -10000.0, 10000.0, f"x_L[{idx}]")
                ensure_range(y_L, -10000.0, 10000.0, f"y_L[{idx}]")
                ensure_range(z_L, 100.0, 10000.0, f"z_L[{idx}]")
                ensure_range(I0, 0.01, 10000.0, f"I0[{idx}]")
                lights.append(LightSource(x=x_L, y=y_L, z=z_L, I0=I0))

            if not lights:
                raise ValueError("Добавьте хотя бы один источник света.")

        except ValueError as exc:
            message = str(exc) or "Проверьте введённые значения."
            messagebox.showerror("Ошибка", message)
            return None

        screen = Screen(W_mm=W_mm, H_mm=H_mm, W_res=W_res, H_res=H_res)
        sphere = Sphere(cx=x_C, cy=y_C, cz=z_C, radius=R_sphere)
        viewer = Viewer(z=z_O)
        shading = ShadingParams(
            k_a=k_a, k_d=k_d, k_s=k_s,
            shininess=shininess, I_ambient=I_amb
        )

        return screen, lights, sphere, viewer, shading

    def _on_calculate(self):
        params = self._read_parameters()
        if params is None:
            return

        screen, lights, sphere, viewer, shading = params

        renderer = SphereRenderer(screen, lights, sphere, viewer, shading)
        renderer.compute()

        renderer.save_image(self.output_path.get())
        renderer.print_control_points()
        renderer.show_image()
        renderer.show_horizontal_section()
        renderer.show_vertical_section()

        messagebox.showinfo("Готово", f"Сохранено: {self.output_path.get()}")


def main():
    app = SphereApp()
    app.mainloop()


if __name__ == "__main__":
    main()
