import tkinter as tk
from PIL import Image, ImageTk

class DisplayHATMini:
    def __init__(self, rotation=0, backlight=True,
                 logical_size=(320, 240), scale=3, title="PiZeroAiCam (VNC)"):
        self.rotation = rotation
        self.w, self.h = logical_size
        self.scale = scale
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas = tk.Canvas(self.root, width=self.w*self.scale, height=self.h*self.scale,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self._photo = None
        self._closed = False
        self.root.update_idletasks(); self.root.update()

    def _prep(self, pil_img):
        img = pil_img.convert("RGB")
        if self.rotation:
            img = img.rotate(self.rotation, expand=True)
        if img.size != (self.w, self.h):
            img = img.resize((self.w, self.h), Image.NEAREST)
        if self.scale != 1:
            img = img.resize((self.w*self.scale, self.h*self.scale), Image.NEAREST)
        return ImageTk.PhotoImage(img)

    def display(self, pil_image):
        if self._closed: return
        self._photo = self._prep(pil_image)
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.root.update_idletasks(); self.root.update()

    set_image = display
    def set_backlight(self, *_, **__): pass
    def set_led(self, *_, **__): pass
    def _on_close(self):
        self._closed = True
        try: self.root.destroy()
        except Exception: pass
