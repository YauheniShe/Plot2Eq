import tkinter as tk


class CanvasPreview(tk.Tk):
    def __init__(self, width=800, height=800, range_x=(-10, 10), range_y=(-10, 10)):
        super().__init__()
        self.title("Символьная регрессия: Тест Канваса [-10, 10]")

        self.canvas_width = width
        self.canvas_height = height
        self.range_x = range_x
        self.range_y = range_y

        # Основной холст для рисования
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack(padx=20, pady=10)

        # Метка для отображения координат
        self.coord_label = tk.Label(
            self, text="Координаты: (0.00, 0.00)", font=("Arial", 12)
        )
        self.coord_label.pack()

        # Кнопка очистки
        self.btn_clear = tk.Button(
            self, text="Очистить график", command=self.clear_canvas, font=("Arial", 12)
        )
        self.btn_clear.pack(pady=10)

        # Отрисовка осей и сетки
        self.draw_grid()

        # Переменные для рисования
        self.last_x = None
        self.last_y = None

        # Биндинги (события мыши)
        self.canvas.bind("<Motion>", self.update_coords)  # Движение мыши
        self.canvas.bind("<Button-1>", self.start_draw)  # Нажатие ЛКМ
        self.canvas.bind("<B1-Motion>", self.draw)  # Движение с зажатой ЛКМ
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)  # Отпускание ЛКМ

    def screen_to_math(self, sx, sy):
        """Переводит пиксели экрана в математические координаты[-10, 10]"""
        mx = self.range_x[0] + (sx / self.canvas_width) * (
            self.range_x[1] - self.range_x[0]
        )
        my = self.range_y[0] + ((self.canvas_height - sy) / self.canvas_height) * (
            self.range_y[1] - self.range_y[0]
        )
        return mx, my

    def math_to_screen(self, mx, my):
        """Переводит математические координаты [-10, 10] в пиксели экрана"""
        sx = (
            (mx - self.range_x[0])
            / (self.range_x[1] - self.range_x[0])
            * self.canvas_width
        )
        sy = (
            self.canvas_height
            - (my - self.range_y[0])
            / (self.range_y[1] - self.range_y[0])
            * self.canvas_height
        )
        return sx, sy

    def draw_grid(self):
        """Отрисовывает сетку, оси и числа"""
        # Вертикальные линии (по X)
        for i in range(self.range_x[0], self.range_x[1] + 1):
            sx, _ = self.math_to_screen(i, 0)
            color = "black" if i == 0 else "#e0e0e0"
            width = 2 if i == 0 else 1
            self.canvas.create_line(
                sx, 0, sx, self.canvas_height, fill=color, width=width
            )

            # Подписи чисел (пропускаем 0, чтобы не накладывалось)
            if i != 0:
                self.canvas.create_text(
                    sx,
                    self.canvas_height / 2 + 12,
                    text=str(i),
                    fill="#888888",
                    font=("Arial", 8),
                )

        # Горизонтальные линии (по Y)
        for i in range(self.range_y[0], self.range_y[1] + 1):
            _, sy = self.math_to_screen(0, i)
            color = "black" if i == 0 else "#e0e0e0"
            width = 2 if i == 0 else 1
            self.canvas.create_line(
                0, sy, self.canvas_width, sy, fill=color, width=width
            )

            if i != 0:
                self.canvas.create_text(
                    self.canvas_width / 2 + 12,
                    sy,
                    text=str(i),
                    fill="#888888",
                    font=("Arial", 8),
                )

    def update_coords(self, event):
        """Обновляет текст с математическими координатами"""
        mx, my = self.screen_to_math(event.x, event.y)
        self.coord_label.config(text=f"Координаты: X={mx:.2f}, Y={my:.2f}")

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Рисуем линию
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="#2196F3",
                width=3,
                capstyle=tk.ROUND,
                smooth=True,
                tags="user_drawing",
            )
            self.last_x, self.last_y = event.x, event.y
            self.update_coords(event)

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Удаляет только то, что нарисовал пользователь"""
        self.canvas.delete("user_drawing")


if __name__ == "__main__":
    app = CanvasPreview()
    app.mainloop()
