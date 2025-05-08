import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

BG_COLOR = "#041A2F"
BTN_BG = "#000011"
BTN_FG = "white"

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.configure(bg=BG_COLOR)
        self.image = None
        self.original_image = None
        self.display_image = None

        # Ana çerçeve (ekranı ikiye böler)
        main_frame = tk.Frame(root, bg=BG_COLOR)
        main_frame.pack(fill="both", expand=True)

        # Sol: Buton paneli + scrollbar
        btn_canvas = tk.Canvas(main_frame, bg=BG_COLOR, highlightthickness=0)
        btn_scroll = tk.Scrollbar(main_frame, orient="vertical", command=btn_canvas.yview)
        btn_frame = tk.Frame(btn_canvas, bg=BG_COLOR)

        btn_frame.bind(
            "<Configure>",
            lambda e: btn_canvas.configure(
                scrollregion=btn_canvas.bbox("all")
            )
        )
        btn_canvas.create_window((0, 0), window=btn_frame, anchor="nw")
        btn_canvas.configure(yscrollcommand=btn_scroll.set, height=600, width=5*150)

        btn_canvas.pack(side="left", fill="y", expand=False)
        btn_scroll.pack(side="left", fill="y")

        btn_width = 16
        btn_height = 2

        # Buton isimleri ve fonksiyonları
        buttons = [
            ("Görsel Aç", self.open_image), ("Kaydet", self.save_image), ("Orjinal Hali", self.restore_original), ("Griye Çevir", self.to_gray), ("RGB Kanalları", self.show_rgb_channels),
            ("Negatif", self.negative), ("Parlaklık +", lambda: self.change_brightness(1.2)), ("Parlaklık -", lambda: self.change_brightness(0.8)), ("Eşikleme", self.threshold), ("Histogram", self.show_histogram),
            ("Histogram Eşitle", self.equalize_histogram), ("Kontrast +", lambda: self.change_contrast(1.5)), ("Kontrast -", lambda: self.change_contrast(0.7)), ("Taşı", self.translate), ("Aynala", self.mirror),
            ("Eğme", self.shear), ("Zoom In", lambda: self.scale(1.2)), ("Zoom Out", lambda: self.scale(0.8)), ("Döndür", self.rotate), ("Kırp", self.crop),
            ("Ortalama Filtre", self.mean_filter), ("Medyan Filtre", self.median_filter), ("Gauss Filtre", self.gaussian_filter), ("Konservatif Filtre", self.conservative_filter), ("Crimmins Speckle", self.crimmins_speckle),
            ("Fourier", self.fourier_transform), ("LPF", self.low_pass_filter), ("HPF", self.high_pass_filter), ("Band Geçiren", self.band_pass_filter), ("Band Durduran", self.band_stop_filter),
            ("Butterworth", self.butterworth_filter), ("Gaussian LPF", self.gaussian_lpf), ("Gaussian HPF", self.gaussian_hpf), ("Homomorfik", self.homomorphic_filter), ("Perspektif Düzelt", self.perspective_transform)
        ]

        for idx, (text, cmd) in enumerate(buttons):
            r, c = divmod(idx, 5)
            tk.Button(
                btn_frame, text=text, command=cmd,
                width=btn_width, height=btn_height,
                bg=BTN_BG, fg=BTN_FG, activebackground="#222244", activeforeground=BTN_FG,
                relief="raised", bd=2
            ).grid(row=r, column=c, padx=2, pady=2, sticky="ew")

        # Sağ: Görsel paneli
        img_frame = tk.Frame(main_frame, bg=BG_COLOR)
        img_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        tk.Label(img_frame, text="Orijinal Hali", bg=BG_COLOR, fg="white", font=("Arial", 12, "bold")).pack(pady=(0,5))
        self.panel_original = tk.Label(img_frame, bg=BG_COLOR)
        self.panel_original.pack(pady=(0,20))
        tk.Label(img_frame, text="İşlenmiş Hali", bg=BG_COLOR, fg="white", font=("Arial", 12, "bold")).pack(pady=(0,5))
        self.panel = tk.Label(img_frame, bg=BG_COLOR)
        self.panel.pack()

        # RGB Sliderları
        rgb_frame = tk.Frame(img_frame, bg=BG_COLOR)
        rgb_frame.pack(pady=20)
        self.r_var = tk.IntVar(value=0)
        self.g_var = tk.IntVar(value=0)
        self.b_var = tk.IntVar(value=0)
        tk.Label(rgb_frame, text="R:", bg=BG_COLOR, fg="white").grid(row=0, column=0)
        tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", variable=self.r_var, command=self.update_rgb, bg=BG_COLOR, fg="white", troughcolor=BTN_BG, highlightthickness=0, length=150).grid(row=0, column=1)
        tk.Label(rgb_frame, text="G:", bg=BG_COLOR, fg="white").grid(row=0, column=2)
        tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", variable=self.g_var, command=self.update_rgb, bg=BG_COLOR, fg="white", troughcolor=BTN_BG, highlightthickness=0, length=150).grid(row=0, column=3)
        tk.Label(rgb_frame, text="B:", bg=BG_COLOR, fg="white").grid(row=0, column=4)
        tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", variable=self.b_var, command=self.update_rgb, bg=BG_COLOR, fg="white", troughcolor=BTN_BG, highlightthickness=0, length=150).grid(row=0, column=5)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.original_image = self.image.copy()
            self.display(self.image)
            self.display_original(self.original_image)
            self.r_var.set(0)
            self.g_var.set(0)
            self.b_var.set(0)

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.image.save(file_path)

    def restore_original(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.display(self.image)
            self.r_var.set(0)
            self.g_var.set(0)
            self.b_var.set(0)

    def display(self, img):
        img = img.copy()
        img.thumbnail((400, 400))
        self.display_image = ImageTk.PhotoImage(img)
        self.panel.config(image=self.display_image)
        self.panel.image = self.display_image

    def display_original(self, img):
        img = img.copy()
        img.thumbnail((400, 400))
        self.display_image_original = ImageTk.PhotoImage(img)
        self.panel_original.config(image=self.display_image_original)
        self.panel_original.image = self.display_image_original

    def to_gray(self):
        if self.image:
            gray = self.image.convert("L")
            self.image = gray
            self.display(self.image)

    def show_rgb_channels(self):
        if self.image:
            r, g, b = self.image.split()
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(r, cmap='Reds')
            plt.title('Red')
            plt.subplot(1,3,2)
            plt.imshow(g, cmap='Greens')
            plt.title('Green')
            plt.subplot(1,3,3)
            plt.imshow(b, cmap='Blues')
            plt.title('Blue')
            plt.show()

    def negative(self):
        if self.image:
            inv = ImageOps.invert(self.image.convert("RGB"))
            self.image = inv
            self.display(self.image)

    def change_brightness(self, factor):
        if self.image:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(factor)
            self.display(self.image)

    def threshold(self):
        if self.image:
            gray = self.image.convert("L")
            arr = np.array(gray)
            thresh = 128
            arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
            self.image = Image.fromarray(arr)
            self.display(self.image)

    def show_histogram(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            plt.hist(arr.flatten(), bins=256, range=[0,256], color='gray')
            plt.title("Histogram")
            plt.show()

    def equalize_histogram(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            eq = cv2.equalizeHist(arr)
            self.image = Image.fromarray(eq)
            self.display(self.image)

    def change_contrast(self, factor):
        if self.image:
            enhancer = ImageEnhance.Contrast(self.image)
            self.image = enhancer.enhance(factor)
            self.display(self.image)

    def translate(self):
        if self.image:
            arr = np.array(self.image)
            M = np.float32([[1, 0, 50], [0, 1, 30]])
            shifted = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]))
            self.image = Image.fromarray(shifted)
            self.display(self.image)

    def mirror(self):
        if self.image:
            mirrored = ImageOps.mirror(self.image)
            self.image = mirrored
            self.display(self.image)

    def shear(self):
        if self.image:
            arr = np.array(self.image)
            rows, cols = arr.shape[:2]
            M = np.float32([[1, 0.5, 0], [0, 1, 0]])
            sheared = cv2.warpAffine(arr, M, (int(cols*1.5), rows))
            self.image = Image.fromarray(sheared)
            self.display(self.image)

    def scale(self, factor):
        if self.image:
            w, h = self.image.size
            new_size = (int(w*factor), int(h*factor))
            scaled = self.image.resize(new_size)
            self.image = scaled
            self.display(self.image)

    def rotate(self):
        if self.image:
            rotated = self.image.rotate(45)
            self.image = rotated
            self.display(self.image)

    def crop(self):
        if self.image:
            w, h = self.image.size
            cropped = self.image.crop((w//4, h//4, w*3//4, h*3//4))
            self.image = cropped
            self.display(self.image)

    def mean_filter(self):
        if self.image:
            arr = np.array(self.image)
            mean = cv2.blur(arr, (5,5))
            self.image = Image.fromarray(mean)
            self.display(self.image)

    def median_filter(self):
        if self.image:
            arr = np.array(self.image)
            median = cv2.medianBlur(arr, 5)
            self.image = Image.fromarray(median)
            self.display(self.image)

    def gaussian_filter(self):
        if self.image:
            arr = np.array(self.image)
            gauss = cv2.GaussianBlur(arr, (5,5), 0)
            self.image = Image.fromarray(gauss)
            self.display(self.image)

    def conservative_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            out = arr.copy()
            for i in range(1, arr.shape[0]-1):
                for j in range(1, arr.shape[1]-1):
                    local = arr[i-1:i+2, j-1:j+2].flatten()
                    local = np.delete(local, 4)
                    if arr[i, j] > local.max():
                        out[i, j] = local.max()
                    elif arr[i, j] < local.min():
                        out[i, j] = local.min()
            self.image = Image.fromarray(out)
            self.display(self.image)

    def crimmins_speckle(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            for _ in range(2):
                for i in range(1, arr.shape[0]-1):
                    for j in range(1, arr.shape[1]-1):
                        local = arr[i-1:i+2, j-1:j+2].flatten()
                        arr[i, j] = np.median(local)
            self.image = Image.fromarray(arr)
            self.display(self.image)

    def fourier_transform(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            magnitude = 20*np.log(np.abs(fshift))
            plt.imshow(magnitude, cmap='gray')
            plt.title("Fourier Magnitude")
            plt.show()

    def low_pass_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols), np.uint8)
            r = 30
            mask[crow-r:crow+r, ccol-r:ccol+r] = 1
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def high_pass_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            mask[crow-r:crow+r, ccol-r:ccol+r] = 0
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def band_pass_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols), np.uint8)
            r_out, r_in = 60, 30
            mask[crow-r_out:crow+r_out, ccol-r_out:ccol+r_out] = 1
            mask[crow-r_in:crow+r_in, ccol-r_in:ccol+r_in] = 0
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def band_stop_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            mask = np.ones((rows, cols), np.uint8)
            r_out, r_in = 60, 30
            mask[crow-r_out:crow+r_out, ccol-r_out:ccol+r_out] = 0
            mask[crow-r_in:crow+r_in, ccol-r_in:ccol+r_in] = 1
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def butterworth_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            n = 2
            D0 = 30
            mask = np.zeros((rows, cols))
            for u in range(rows):
                for v in range(cols):
                    D = np.sqrt((u-crow)**2 + (v-ccol)**2)
                    mask[u, v] = 1 / (1 + (D/D0)**(2*n))
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def gaussian_lpf(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            sigma = 30
            x = np.arange(0, rows)
            y = np.arange(0, cols)
            X, Y = np.meshgrid(x, y)
            mask = np.exp(-((X-crow)**2 + (Y-ccol)**2) / (2*sigma**2))
            fshift = fshift * mask.T
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def gaussian_hpf(self):
        if self.image:
            arr = np.array(self.image.convert("L"))
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            sigma = 30
            x = np.arange(0, rows)
            y = np.arange(0, cols)
            X, Y = np.meshgrid(x, y)
            mask = 1 - np.exp(-((X-crow)**2 + (Y-ccol)**2) / (2*sigma**2))
            fshift = fshift * mask.T
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def homomorphic_filter(self):
        if self.image:
            arr = np.array(self.image.convert("L")).astype(np.float32)
            arr = np.log1p(arr)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            f = np.fft.fft2(arr)
            fshift = np.fft.fftshift(f)
            gammaL = 0.5
            gammaH = 2.0
            c = 1
            D0 = 30
            x = np.arange(0, rows)
            y = np.arange(0, cols)
            X, Y = np.meshgrid(x, y)
            D2 = (X-crow)**2 + (Y-ccol)**2
            H = (gammaH - gammaL)*(1 - np.exp(-c*D2/(D0**2))) + gammaL
            fshift = fshift * H.T
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.expm1(np.abs(img_back))
            img_back = np.clip(img_back, 0, 255)
            self.image = Image.fromarray(img_back.astype(np.uint8))
            self.display(self.image)

    def perspective_transform(self):
        if self.image:
            arr = np.array(self.image)
            rows, cols = arr.shape[:2]
            pts1 = np.float32([[50,50],[cols-50,50],[50,rows-50],[cols-50,rows-50]])
            pts2 = np.float32([[10,100],[cols-100,50],[100,rows-100],[cols-50,rows-50]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(arr, M, (cols, rows))
            self.image = Image.fromarray(dst)
            self.display(self.image)

    def update_rgb(self, event=None):
        if self.image and self.original_image:
            arr = np.array(self.original_image.convert("RGB")).astype(np.int16)
            r, g, b = self.r_var.get(), self.g_var.get(), self.b_var.get()
            arr[..., 0] = np.clip(arr[..., 0] + r, 0, 255)
            arr[..., 1] = np.clip(arr[..., 1] + g, 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] + b, 0, 255)
            self.image = Image.fromarray(arr.astype(np.uint8))
            self.display(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()