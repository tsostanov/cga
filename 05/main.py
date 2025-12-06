import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List, Tuple
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
        assert len(spheres) == 2, "Нужно передать ровно две сферы"
        self.screen = screen
        self.lights = lights
        self.spheres = spheres
        self.viewer = viewer
        self.shading = shading

        self.rgb_img = None
        self.x_vals = None
        self.y_vals = None

    @staticmethod
    def _normalize(vx, vy, vz):
        r = np.sqrt(vx * vx + vy * vy + vz * vz)
        r[r == 0] = 1e-6
        return vx / r, vy / r, vz / r

    @staticmethod
    def _ray_sphere_intersection(Px, Py, Pz,
                                 Lx, Ly, Lz,
                                 sphere: Sphere):
        Dx = Lx - Px
        Dy = Ly - Py
        Dz = Lz - Pz

        Ox = Px - sphere.cx
        Oy = Py - sphere.cy
        Oz = Pz - sphere.cz

        a = Dx * Dx + Dy * Dy + Dz * Dz
        b = 2.0 * (Ox * Dx + Oy * Dy + Oz * Dz)
        c = Ox * Ox + Oy * Oy + Oz * Oz - sphere.radius ** 2

        disc = b * b - 4.0 * a * c
        mask_disc = disc > 0.0
        if not np.any(mask_disc):
            return np.zeros_like(Px, dtype=bool)

        sqrt_disc = np.zeros_like(Px)
        sqrt_disc[mask_disc] = np.sqrt(disc[mask_disc])

        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        eps = 1e-4
        hit1 = (t1 > eps) & (t1 < 1.0 - eps)
        hit2 = (t2 > eps) & (t2 < 1.0 - eps)
        return hit1 | hit2

    def compute(self):
        W_mm = self.screen.W_mm
        H_mm = self.screen.H_mm
        W_res = self.screen.W_res
        H_res = self.screen.H_res

        x_min, x_max = -W_mm / 2.0, W_mm / 2.0
        y_min, y_max = -H_mm / 2.0, H_mm / 2.0

        x_vals = np.linspace(x_min, x_max, W_res)
        y_vals = np.linspace(y_min, y_max, H_res)
        X, Y = np.meshgrid(x_vals, y_vals)

        self.x_vals = x_vals
        self.y_vals = y_vals

        img = np.zeros((H_res, W_res, 3), dtype=float)
        z_buffer = np.full((H_res, W_res), np.inf, dtype=float)

        k_a = self.shading.k_a
        k_d = self.shading.k_d
        k_s = self.shading.k_s
        n = self.shading.shininess
        I_amb = self.shading.I_ambient
        z_O = self.viewer.z

        for s_idx, sphere in enumerate(self.spheres):
            dx = X - sphere.cx
            dy = Y - sphere.cy
            rho2 = dx * dx + dy * dy
            R2 = sphere.radius ** 2

            inside = rho2 <= R2
            if not np.any(inside):
                continue

            z_offset = np.zeros_like(X)
            z_offset[inside] = np.sqrt(R2 - rho2[inside])

            if z_O < sphere.cz:
                Z = sphere.cz - z_offset
            else:
                Z = sphere.cz + z_offset

            dist_to_viewer = np.abs(Z - z_O)
            visible = inside & (dist_to_viewer < z_buffer)

            if not np.any(visible):
                continue

            z_buffer[visible] = dist_to_viewer[visible]

            Px = X[visible]
            Py = Y[visible]
            Pz = Z[visible]

            Nx = (Px - sphere.cx) / sphere.radius
            Ny = (Py - sphere.cy) / sphere.radius
            Nz = (Pz - sphere.cz) / sphere.radius

            Vx = -Px
            Vy = -Py
            Vz = z_O - Pz
            Vx, Vy, Vz = self._normalize(Vx, Vy, Vz)

            C_sphere = np.array(sphere.color, dtype=float)
            pixel_color = (k_a * I_amb) * np.tile(C_sphere, (Px.size, 1))

            for light in self.lights:
                Lx = light.x - Px
                Ly = light.y - Py
                Lz = light.z - Pz
                r_L = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
                r_L[r_L == 0] = 1e-6
                Lx_n = Lx / r_L
                Ly_n = Ly / r_L
                Lz_n = Lz / r_L

                Hx = Lx_n + Vx
                Hy = Ly_n + Vy
                Hz = Lz_n + Vz
                Hx, Hy, Hz = self._normalize(Hx, Hy, Hz)

                N_dot_L = Nx * Lx_n + Ny * Ly_n + Nz * Lz_n
                N_dot_L = np.clip(N_dot_L, 0.0, None)

                N_dot_H = Nx * Hx + Ny * Hy + Nz * Hz
                N_dot_H = np.clip(N_dot_H, 0.0, None)

                attenuation = light.I0 / (r_L ** 2)

                shadow_mask = np.zeros_like(Px, dtype=bool)
                for other_idx, other_sphere in enumerate(self.spheres):
                    if other_idx == s_idx:
                        continue
                    shadow_here = self._ray_sphere_intersection(
                        Px, Py, Pz,
                        light.x * np.ones_like(Px),
                        light.y * np.ones_like(Py),
                        light.z * np.ones_like(Pz),
                        other_sphere,
                    )
                    shadow_mask |= shadow_here

                visibility = (~shadow_mask).astype(float)

                I_diff = k_d * attenuation * N_dot_L * visibility
                I_spec = k_s * attenuation * (N_dot_H ** n) * visibility

                light_color = np.array(light.color, dtype=float)

                pixel_color += (
                    (I_diff[:, None] * C_sphere[None, :] +
                     I_spec[:, None]) * light_color[None, :]
                )

            img[visible, :] = pixel_color

        I_max = img.max()
        if I_max <= 0:
            I_max = 1.0
        img_norm = np.clip(img / I_max * 255.0, 0, 255).astype(np.uint8)
        self.rgb_img = img_norm
        return img_norm

    def save_image(self, filename: str):
        if self.rgb_img is None:
            return
        Image.fromarray(self.rgb_img, mode="RGB").save(filename)

    def show_image(self):
        if self.rgb_img is None:
            return
        x_min, x_max = self.x_vals[0], self.x_vals[-1]
        y_min, y_max = self.y_vals[0], self.y_vals[-1]

        plt.figure()
        plt.imshow(self.rgb_img, extent=[x_min, x_max, y_max, y_min])
        plt.title("Распределение яркости на сферах")
        plt.xlabel("x, мм")
        plt.ylabel("y, мм")
        plt.tight_layout()
        plt.show()

    def show_scene_projections(self):
        s1, s2 = self.spheres
        lights = self.lights
        viewer = self.viewer

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        ax = axes[0]
        for s, col in zip(self.spheres, ["tab:blue", "tab:orange"]):
            circ = plt.Circle((s.cx, s.cy), s.radius, fill=False, color=col)
            ax.add_patch(circ)
            ax.plot(s.cx, s.cy, "o", color=col)
        for i, L in enumerate(lights):
            ax.plot(L.x, L.y, "*", markersize=8, label=f"LS{i+1}")
        ax.plot(0, 0, "^", label="Viewer")
        ax.set_xlabel("X, мм")
        ax.set_ylabel("Y, мм")
        ax.set_title("Проекция XY")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)

        ax = axes[1]
        for s, col in zip(self.spheres, ["tab:blue", "tab:orange"]):
            circ = plt.Circle((s.cx, s.cz), s.radius, fill=False, color=col)
            ax.add_patch(circ)
            ax.plot(s.cx, s.cz, "o", color=col)
        for i, L in enumerate(lights):
            ax.plot(L.x, L.z, "*", markersize=8, label=f"LS{i+1}")
        ax.plot(0, viewer.z, "^", label="Viewer")
        ax.set_xlabel("X, мм")
        ax.set_ylabel("Z, мм")
        ax.set_title("Проекция XZ")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)

        ax = axes[2]
        for s, col in zip(self.spheres, ["tab:blue", "tab:orange"]):
            circ = plt.Circle((s.cy, s.cz), s.radius, fill=False, color=col)
            ax.add_patch(circ)
            ax.plot(s.cy, s.cz, "o", color=col)
        for i, L in enumerate(lights):
            ax.plot(L.y, L.z, "*", markersize=8, label=f"LS{i+1}")
        ax.plot(0, viewer.z, "^", label="Viewer")
        ax.set_xlabel("Y, мм")
        ax.set_ylabel("Z, мм")
        ax.set_title("Проекция YZ")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


class TwoSpheresApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР-5: яркость, цвета, тени (Блинн-Фонг)")

        self.entries = {}
        self.light_entries = []
        self.light_row_widgets = []
        self.light_frame = None
        self.output_path = tk.StringVar(value="two_spheres_color_shadow.png")
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
            defaults = {
                "x": 0.0, "y": 0.0, "z": 500.0, "I0": 1000.0,
                "R": 1.0, "G": 1.0, "B": 1.0
            }
        if self.light_frame is None:
            return
        row_index = len(self.light_entries) + 1
        vars_row = {}
        row_widgets = []
        for col, key in enumerate(("x", "y", "z", "I0", "R", "G", "B")):
            value = defaults.get(key, 0.0)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(self.light_frame, textvariable=var, width=8)
            entry.grid(row=row_index, column=col, padx=3, pady=2)
            row_widgets.append(entry)
            vars_row[key] = var
        self.light_entries.append(vars_row)
        self.light_row_widgets.append(row_widgets)

    def _remove_light_row(self):
        if not self.light_entries:
            return
        if len(self.light_entries) == 1:
            messagebox.showwarning("Внимание", "Должен остаться хотя бы один источник света.")
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
        for col, text in enumerate(
                ("x_L (мм)", "y_L (мм)", "z_L (мм)", "I0", "R_L", "G_L", "B_L")):
            tk.Label(self.light_frame, text=text, anchor="w").grid(
                row=0, column=col, padx=3, pady=2
            )
        self._add_light_row()
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=row + 1, column=0, columnspan=3, pady=5)
        ttk.Button(controls_frame, text="Добавить источник",
                   command=self._add_light_row).grid(row=0, column=0, padx=5)
        ttk.Button(controls_frame, text="Убрать источник",
                   command=self._remove_light_row).grid(row=0, column=1, padx=5)
        row += 2

        ttk.Label(self, text="Сфера 1").grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        self._add_field(row, "x_C1", -200.0)
        row += 1
        self._add_field(row, "y_C1", 0.0)
        row += 1
        self._add_field(row, "z_C1", 2000.0)
        row += 1
        self._add_field(row, "R1", 800.0)
        row += 1
        self._add_field(row, "Rs1", 0.2)
        row += 1
        self._add_field(row, "Gs1", 0.8)
        row += 1
        self._add_field(row, "Bs1", 0.2)
        row += 1

        ttk.Label(self, text="Сфера 2").grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        self._add_field(row, "x_C2", 300.0)
        row += 1
        self._add_field(row, "y_C2", 200.0)
        row += 1
        self._add_field(row, "z_C2", 2200.0)
        row += 1
        self._add_field(row, "R2", 400.0)
        row += 1
        self._add_field(row, "Rs2", 0.9)
        row += 1
        self._add_field(row, "Gs2", 0.7)
        row += 1
        self._add_field(row, "Bs2", 0.2)
        row += 1

        self._add_field(row, "z_O", 0.0)
        row += 1

        self._add_field(row, "k_a", 0.1)
        row += 1
        self._add_field(row, "k_d", 1.0)
        row += 1
        self._add_field(row, "k_s", 0.6)
        row += 1
        self._add_field(row, "shininess", 50.0)
        row += 1
        self._add_field(row, "I_ambient", 1.0)
        row += 1

        tk.Label(self, text="Файл вывода").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(self, textvariable=self.output_path, width=25).grid(
            row=row, column=1, padx=5, pady=2
        )
        ttk.Button(self, text="...", command=self._browse_file).grid(
            row=row, column=2, padx=5, pady=2
        )
        row += 1

        status_label = tk.Label(self, textvariable=self.status_var,
                                anchor="w", fg="gray")
        status_label.grid(row=row, column=0, columnspan=3,
                          sticky="we", padx=5, pady=2)
        row += 1

        ttk.Button(self, text="Вычислить и показать",
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
                raise ValueError(f"{name} выходит за пределы [{min_value}, {max_value}].")

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
                f"Ширина пересчитана: {W_res} пикс."
            )

            x_C1 = float(self.entries["x_C1"].get())
            y_C1 = float(self.entries["y_C1"].get())
            z_C1 = float(self.entries["z_C1"].get())
            R1 = float(self.entries["R1"].get())

            ensure_range(x_C1, -10000.0, 10000.0, "x_C1")
            ensure_range(y_C1, -10000.0, 10000.0, "y_C1")
            ensure_range(z_C1, 100.0, 10000.0, "z_C1")
            if R1 <= 0:
                raise ValueError("R1 должен быть > 0.")

            Rs1 = float(self.entries["Rs1"].get())
            Gs1 = float(self.entries["Gs1"].get())
            Bs1 = float(self.entries["Bs1"].get())

            x_C2 = float(self.entries["x_C2"].get())
            y_C2 = float(self.entries["y_C2"].get())
            z_C2 = float(self.entries["z_C2"].get())
            R2 = float(self.entries["R2"].get())

            ensure_range(x_C2, -10000.0, 10000.0, "x_C2")
            ensure_range(y_C2, -10000.0, 10000.0, "y_C2")
            ensure_range(z_C2, 100.0, 10000.0, "z_C2")
            if R2 <= 0:
                raise ValueError("R2 должен быть > 0.")

            Rs2 = float(self.entries["Rs2"].get())
            Gs2 = float(self.entries["Gs2"].get())
            Bs2 = float(self.entries["Bs2"].get())

            for name, val in [
                ("Rs1", Rs1), ("Gs1", Gs1), ("Bs1", Bs1),
                ("Rs2", Rs2), ("Gs2", Gs2), ("Bs2", Bs2),
            ]:
                ensure_range(val, 0.0, 1.0, name)

            z_O = float(self.entries["z_O"].get())
            ensure_range(z_O, 0.0, 10000.0, "z_O")

            k_a = float(self.entries["k_a"].get())
            k_d = float(self.entries["k_d"].get())
            k_s = float(self.entries["k_s"].get())
            shininess = float(self.entries["shininess"].get())
            I_amb = float(self.entries["I_ambient"].get())

            for name, value in (
                    ("k_a", k_a), ("k_d", k_d),
                    ("k_s", k_s), ("I_ambient", I_amb)):
                if value < 0:
                    raise ValueError(f"{name} должен быть >= 0.")
            if shininess <= 0:
                raise ValueError("shininess должен быть > 0.")

            lights = []
            for idx, vars_row in enumerate(self.light_entries, start=1):
                values = {key: vars_row[key].get().strip()
                          for key in ("x", "y", "z", "I0", "R", "G", "B")}
                if not any(values.values()):
                    continue

                x_L = float(values["x"])
                y_L = float(values["y"])
                z_L = float(values["z"])
                I0 = float(values["I0"])
                R_L = float(values["R"])
                G_L = float(values["G"])
                B_L = float(values["B"])

                ensure_range(x_L, -10000.0, 10000.0, f"x_L[{idx}]")
                ensure_range(y_L, -10000.0, 10000.0, f"y_L[{idx}]")
                ensure_range(z_L, 100.0, 10000.0, f"z_L[{idx}]")
                ensure_range(I0, 0.01, 10000.0, f"I0[{idx}]")
                for name, val in [
                    (f"R_L[{idx}]", R_L),
                    (f"G_L[{idx}]", G_L),
                    (f"B_L[{idx}]", B_L),
                ]:
                    ensure_range(val, 0.0, 1.0, name)

                lights.append(
                    LightSource(x=x_L, y=y_L, z=z_L,
                                I0=I0, color=(R_L, G_L, B_L))
                )

            if not lights:
                raise ValueError("Нужно указать хотя бы один источник света.")

        except ValueError as exc:
            message = str(exc) or "Ошибка ввода параметров."
            messagebox.showerror("Ошибка", message)
            return None

        screen = Screen(W_mm=W_mm, H_mm=H_mm, W_res=W_res, H_res=H_res)
        sphere1 = Sphere(
            cx=x_C1, cy=y_C1, cz=z_C1, radius=R1,
            color=(Rs1, Gs1, Bs1)
        )
        sphere2 = Sphere(
            cx=x_C2, cy=y_C2, cz=z_C2, radius=R2,
            color=(Rs2, Gs2, Bs2)
        )
        viewer = Viewer(z=z_O)
        shading = ShadingParams(
            k_a=k_a, k_d=k_d, k_s=k_s,
            shininess=shininess, I_ambient=I_amb
        )

        return screen, lights, [sphere1, sphere2], viewer, shading

    def _on_calculate(self):
        params = self._read_parameters()
        if params is None:
            return

        screen, lights, spheres, viewer, shading = params

        renderer = TwoSpheresRenderer(screen, lights, spheres, viewer, shading)
        renderer.compute()
        renderer.save_image(self.output_path.get())
        renderer.show_image()
        renderer.show_scene_projections()

        messagebox.showinfo("Готово", f"Сохранено: {self.output_path.get()}")


def main():
    app = TwoSpheresApp()
    app.mainloop()


if __name__ == "__main__":
    main()
