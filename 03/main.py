import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def compute_illumination(
    W_mm,
    H_mm,
    W_res,
    H_res,
    x_L,
    y_L,
    z_L,
    I0,
    circle_cx,
    circle_cy,
    circle_r,
):
    x_min, x_max = -W_mm / 2, W_mm / 2
    y_min, y_max = -H_mm / 2, H_mm / 2

    x_vals = np.linspace(x_min, x_max, W_res)
    y_vals = np.linspace(y_min, y_max, H_res)
    X, Y = np.meshgrid(x_vals, y_vals)

    R = np.sqrt((X - x_L) ** 2 + (Y - y_L) ** 2 + z_L ** 2)
    R[R == 0] = 1e-6

    E = I0 * z_L / (R ** 3)

    dist_to_center = np.sqrt((X - circle_cx) ** 2 + (Y - circle_cy) ** 2)
    mask_circle = dist_to_center <= circle_r

    E_masked = np.where(mask_circle, E, 0.0)

    E_inside = E[mask_circle]
    if E_inside.size == 0:
        E_max = 1.0
    else:
        E_max = np.max(E_inside)
        if E_max == 0:
            E_max = 1.0

    E_norm = (E_masked / E_max) * 255.0
    E_img = np.clip(E_norm, 0, 255).astype(np.uint8)

    return E_img, E, x_vals, y_vals, mask_circle


def save_image(img_array, filename):
    Image.fromarray(img_array, mode="L").save(filename)


def show_image(img_array, x_vals, y_vals, circle_cx, circle_cy, circle_r):
    x_min, x_max = x_vals[0], x_vals[-1]
    y_min, y_max = y_vals[0], y_vals[-1]

    plt.figure()
    plt.imshow(img_array, cmap="gray", extent=[x_min, x_max, y_max, y_min])
    circle = plt.Circle((circle_cx, circle_cy), circle_r, color="red", fill=False, linewidth=1)
    plt.gca().add_patch(circle)
    plt.title("Распределение освещённости")
    plt.xlabel("x, мм")
    plt.ylabel("y, мм")
    plt.tight_layout()
    plt.show()


def show_section(E_masked, x_vals, y_vals, center_y):
    row_idx = (np.abs(y_vals - center_y)).argmin()
    section_x = x_vals
    section_E = E_masked[row_idx, :]

    plt.figure()
    plt.plot(section_x, section_E)
    plt.title("Сечение через центр")
    plt.xlabel("x, мм")
    plt.ylabel("E")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def illumination_at_point(x, y, x_L, y_L, z_L, I0):
    r = np.sqrt((x - x_L) ** 2 + (y - y_L) ** 2 + z_L ** 2)
    return I0 * z_L / (r ** 3)


def print_control_points(circle_cx, circle_cy, circle_r, x_L, y_L, z_L, I0, E, mask_circle):
    print("\n--- Контрольные точки ---")
    points = {
        "center": (circle_cx, circle_cy),
        "+X": (circle_cx + circle_r, circle_cy),
        "-X": (circle_cx - circle_r, circle_cy),
        "+Y": (circle_cx, circle_cy + circle_r),
        "-Y": (circle_cx, circle_cy - circle_r),
    }
    for name, (px, py) in points.items():
        val = illumination_at_point(px, py, x_L, y_L, z_L, I0)
        print(f"{name:6s}: x={px:8.2f} y={py:8.2f} -> E={val:.6f}")

    E_inside = E[mask_circle]
    if E_inside.size > 0:
        print("\n--- Характеристики по кругу ---")
        print(f"E_max = {np.max(E_inside):.6f}")
        print(f"E_min = {np.min(E_inside):.6f}")
        print(f"E_avg = {np.mean(E_inside):.6f}")
    else:
        print("\n(круг вне области)")


def main():
    root = tk.Tk()
    root.title("Освещённость на плоскости")

    entries = {}

    def add_field(row, label, default):
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(root, textvariable=var, width=15)
        ent.grid(row=row, column=1, padx=5, pady=2)
        entries[label] = var

    add_field(0, "W_mm", 1000)
    add_field(1, "H_mm", 1000)
    add_field(2, "W_res", 400)
    add_field(3, "H_res", 400)
    add_field(4, "x_L", 0.0)
    add_field(5, "y_L", 0.0)
    add_field(6, "z_L", 500.0)
    add_field(7, "I0", 1000.0)
    add_field(8, "circle_cx", 0.0)
    add_field(9, "circle_cy", 0.0)
    add_field(10, "circle_r", 400.0)

    output_path = tk.StringVar(value="illumination.png")

    tk.Label(root, text="Файл").grid(row=11, column=0, sticky="w", padx=5, pady=2)
    ttk.Entry(root, textvariable=output_path, width=25).grid(row=11, column=1, padx=5, pady=2)

    def browse_file():
        fname = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fname:
            output_path.set(fname)

    ttk.Button(root, text="...", command=browse_file).grid(row=11, column=2, padx=5, pady=2)

    def run_calculation():
        try:
            W_mm_v = float(entries["W_mm"].get())
            H_mm_v = float(entries["H_mm"].get())
            W_res_v = int(entries["W_res"].get())
            H_res_v = int(entries["H_res"].get())
            x_L_v = float(entries["x_L"].get())
            y_L_v = float(entries["y_L"].get())
            z_L_v = float(entries["z_L"].get())
            I0_v = float(entries["I0"].get())
            circle_cx_v = float(entries["circle_cx"].get())
            circle_cy_v = float(entries["circle_cy"].get())
            circle_r_v = float(entries["circle_r"].get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте введённые значения.")
            return

        E_img, E_raw, x_vals, y_vals, mask_circle = compute_illumination(
            W_mm_v,
            H_mm_v,
            W_res_v,
            H_res_v,
            x_L_v,
            y_L_v,
            z_L_v,
            I0_v,
            circle_cx_v,
            circle_cy_v,
            circle_r_v,
        )

        save_image(E_img, output_path.get())

        print_control_points(
            circle_cx_v,
            circle_cy_v,
            circle_r_v,
            x_L_v,
            y_L_v,
            z_L_v,
            I0_v,
            E_raw,
            mask_circle,
        )

        show_image(E_img, x_vals, y_vals, circle_cx_v, circle_cy_v, circle_r_v)
        show_section(E_raw * mask_circle, x_vals, y_vals, circle_cy_v)

        messagebox.showinfo("Готово", f"Сохранено: {output_path.get()}")

    ttk.Button(root, text="Рассчитать и визуализировать", command=run_calculation).grid(
        row=12, column=0, columnspan=3, pady=10
    )

    root.mainloop()


if __name__ == "__main__":
    main()
