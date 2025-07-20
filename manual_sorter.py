import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk

'''Ручная сортировка изображений по 2 классам'''

class ImageSorter:
    def __init__(self, root, source_folder):
        self.root = root
        self.source_folder = source_folder
        self.low_folder = os.path.join(source_folder, "low")
        self.high_folder = os.path.join(source_folder, "high")

        # Создаем папки для сортировки, если их нет
        os.makedirs(self.low_folder, exist_ok=True)
        os.makedirs(self.high_folder, exist_ok=True)

        # Собираем все изображения из подпапок
        self.image_paths = []
        for subfolder in sorted(os.listdir(source_folder)):
            subfolder_path = os.path.join(source_folder, subfolder)
            if os.path.isdir(subfolder_path) and subfolder.isdigit():
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_paths.append({
                            'subfolder': subfolder,
                            'path': os.path.join(subfolder_path, file),
                            'filename': file
                        })

        self.current_index = 0
        self.setup_ui()
        self.show_image()

    def setup_ui(self):
        self.root.title("Image Sorter")
        self.root.geometry("1000x800")

        # Frame для изображения с прокруткой
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Label для изображения
        self.image_label = tk.Label(self.scrollable_frame)
        self.image_label.pack(pady=20)

        # Label с информацией
        self.info_label = tk.Label(self.scrollable_frame, text="", font=('Arial', 14))
        self.info_label.pack(pady=10)

        # Frame для кнопок
        self.button_frame = tk.Frame(self.scrollable_frame)
        self.button_frame.pack(pady=20)

        # Кнопки управления
        tk.Button(self.button_frame, text="1 - Low", command=self.move_to_low,
                  width=15, height=2, font=('Arial', 12)).grid(row=0, column=0, padx=15)
        tk.Button(self.button_frame, text="2 - High", command=self.move_to_high,
                  width=15, height=2, font=('Arial', 12)).grid(row=0, column=1, padx=15)
        tk.Button(self.button_frame, text="3 - Skip", command=self.skip_image,
                  width=15, height=2, font=('Arial', 12)).grid(row=0, column=2, padx=15)

        # Привязка клавиш
        self.root.bind('1', lambda e: self.move_to_low())
        self.root.bind('2', lambda e: self.move_to_high())
        self.root.bind('3', lambda e: self.skip_image())

        # Кнопка для выхода
        tk.Button(self.scrollable_frame, text="Exit", command=self.root.quit,
                  width=15, height=2, font=('Arial', 12)).pack(pady=20)

    def show_image(self):
        if self.current_index >= len(self.image_paths):
            self.info_label.config(text="Все изображения обработаны!")
            self.image_label.config(image=None)
            return

        current_image = self.image_paths[self.current_index]
        self.info_label.config(
            text=f"Из папки: {current_image['subfolder']}\n"
                 f"Изображение {self.current_index + 1} из {len(self.image_paths)}\n"
                 f"Имя файла: {current_image['filename']}")

        try:
            image = Image.open(current_image['path'])
            original_width, original_height = image.size
            new_size = (original_width * 2, original_height * 2)
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.canvas.yview_moveto(0)
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            self.skip_image()

    def move_image(self, target_folder):
        if self.current_index >= len(self.image_paths):
            return

        current_image = self.image_paths[self.current_index]
        new_filename = f"{current_image['subfolder']}_{current_image['filename']}"
        destination = os.path.join(target_folder, new_filename)

        try:
            shutil.move(current_image['path'], destination)
            print(f"Перемещено в {target_folder}: {current_image['filename']}")
            self.current_index += 1
            self.show_image()
        except Exception as e:
            print(f"Ошибка перемещения файла: {e}")

    def move_to_low(self):
        self.move_image(self.low_folder)

    def move_to_high(self):
        self.move_image(self.high_folder)

    def skip_image(self):
        self.current_index += 1
        self.show_image()


if __name__ == "__main__":
    source_directory = "глобальный/путь"
    root = tk.Tk()
    app = ImageSorter(root, source_directory)
    root.mainloop()
