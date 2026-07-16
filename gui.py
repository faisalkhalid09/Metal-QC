"""
gui.py — Tkinter GUI Module
============================
Modernized UI with ttk, Menu bar, Toolbar, thumbnails,
status bar, progress bar, zoom/pan canvas, and keyboard shortcuts.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import logging

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

from align import align_images
from diff import detect_differences
from visualize import build_composite

logger = logging.getLogger(__name__)


class QCApp(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.title("Metal Assembly Visual QC")
        
        # Apply a basic ttk theme (clam is cross-platform and looks better than default Windows)
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        self.geometry("1200x800")
        self.minsize(900, 600)

        self._path_a: str = ""
        self._path_b: str = ""
        self._composite: np.ndarray | None = None
        self._pil_img: Image.Image | None = None
        self._zoom_factor = 1.0

        self._build_ui()
        self._bind_shortcuts()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # 1. Menu Bar
        menubar = tk.Menu(self)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image A...", command=self._select_a, accelerator="Ctrl+1")
        file_menu.add_command(label="Open Image B...", command=self._select_b, accelerator="Ctrl+2")
        file_menu.add_separator()
        file_menu.add_command(label="Save Result...", command=self._save_result, accelerator="Ctrl+S", state=tk.DISABLED)
        self.file_menu = file_menu
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)

        # 2. Toolbar
        toolbar = ttk.Frame(self, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self._btn_a = ttk.Button(toolbar, text="Open Image A", command=self._select_a)
        self._btn_a.pack(side=tk.LEFT, padx=2, pady=2)
        
        self._btn_b = ttk.Button(toolbar, text="Open Image B", command=self._select_b)
        self._btn_b.pack(side=tk.LEFT, padx=2, pady=2)
        
        self._btn_compare = ttk.Button(toolbar, text="Compare", command=self._run_compare, state=tk.DISABLED)
        self._btn_compare.pack(side=tk.LEFT, padx=10, pady=2)
        
        self._btn_save = ttk.Button(toolbar, text="Save", command=self._save_result, state=tk.DISABLED)
        self._btn_save.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Zoom controls
        self._btn_zoom_in = ttk.Button(toolbar, text="Zoom In (+)", command=self._zoom_in)
        self._btn_zoom_in.pack(side=tk.RIGHT, padx=2, pady=2)
        
        self._btn_zoom_out = ttk.Button(toolbar, text="Zoom Out (-)", command=self._zoom_out)
        self._btn_zoom_out.pack(side=tk.RIGHT, padx=2, pady=2)

        self._btn_fit = ttk.Button(toolbar, text="Fit to Screen", command=self._fit_to_screen)
        self._btn_fit.pack(side=tk.RIGHT, padx=2, pady=2)

        # 3. Main PanedWindow (Thumbnails on left, Result on right)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel: Thumbnails
        left_panel = ttk.Frame(paned, width=250)
        paned.add(left_panel, weight=0)
        
        ttk.Label(left_panel, text="Reference Image A", font=("Segoe UI", 9, "bold")).pack(pady=(5,0))
        self._lbl_thumb_a = tk.Label(left_panel, bg="#333", width=30, height=10)
        self._lbl_thumb_a.pack(padx=10, pady=5, fill=tk.X)
        self._lbl_path_a = ttk.Label(left_panel, text="None", wraplength=200, justify=tk.CENTER)
        self._lbl_path_a.pack(padx=10, pady=(0, 15))
        
        ttk.Label(left_panel, text="Inspection Image B", font=("Segoe UI", 9, "bold")).pack(pady=(5,0))
        self._lbl_thumb_b = tk.Label(left_panel, bg="#333", width=30, height=10)
        self._lbl_thumb_b.pack(padx=10, pady=5, fill=tk.X)
        self._lbl_path_b = ttk.Label(left_panel, text="None", wraplength=200, justify=tk.CENTER)
        self._lbl_path_b.pack(padx=10, pady=5)

        # Summary Panel
        summary_frame = ttk.LabelFrame(left_panel, text="Analysis Summary")
        summary_frame.pack(padx=10, pady=10, fill=tk.X)
        self._lbl_summary = ttk.Label(summary_frame, text="No comparison run.", justify=tk.LEFT, wraplength=200)
        self._lbl_summary.pack(padx=5, pady=5, anchor=tk.W)

        # Isolated Diff View
        self._lbl_diff_thumb = tk.Label(summary_frame, bg="#333", width=30, height=8)
        self._lbl_diff_thumb.pack(padx=5, pady=5, fill=tk.X)

        # Right Panel: Canvas
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)
        
        self._canvas = tk.Canvas(right_panel, bg="#222", cursor="crosshair")
        h_scroll = ttk.Scrollbar(right_panel, orient=tk.HORIZONTAL, command=self._canvas.xview)
        v_scroll = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for pan/zoom
        self._canvas.bind("<ButtonPress-1>", self._start_pan)
        self._canvas.bind("<B1-Motion>", self._do_pan)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

        # 4. Status Bar & Progress
        status_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self._progress = ttk.Progressbar(status_frame, mode="indeterminate", length=150)
        self._progress.pack(side=tk.RIGHT, padx=5, pady=2)
        
        self._status_var = tk.StringVar(value="Ready — select both images to begin.")
        status_lbl = ttk.Label(status_frame, textvariable=self._status_var, anchor="w")
        status_lbl.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=2)

    def _bind_shortcuts(self):
        self.bind("<Control-Key-1>", lambda e: self._select_a())
        self.bind("<Control-Key-2>", lambda e: self._select_b())
        self.bind("<Control-Key-s>", lambda e: self._save_result())
        self.bind("<Control-Key-q>", lambda e: self.quit())
        self.bind("<Control-plus>", lambda e: self._zoom_in())
        self.bind("<Control-equal>", lambda e: self._zoom_in())
        self.bind("<Control-minus>", lambda e: self._zoom_out())
        self.bind("<Control-0>", lambda e: self._fit_to_screen())

    def _show_about(self):
        messagebox.showinfo("About", "Metal Assembly Visual QC\nVersion 1.0\n\nCompares two metal assembly photos and highlights differences.")

    # -----------------------------------------------------------------------
    # File selection & Thumbnails
    # -----------------------------------------------------------------------

    def _load_thumbnail(self, path: str, label: tk.Label):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, bg="black")
            label.image = photo  # Keep reference
        except Exception as e:
            logger.error("Thumbnail load error: %s", e)

    def _select_a(self):
        path = filedialog.askopenfilename(
            title="Select Reference Image A",
            filetypes=[("JPEG/PNG images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if path:
            self._path_a = path
            self._lbl_path_a.config(text=os.path.basename(path))
            self._load_thumbnail(path, self._lbl_thumb_a)
            self._update_compare_btn()

    def _select_b(self):
        path = filedialog.askopenfilename(
            title="Select Inspection Image B",
            filetypes=[("JPEG/PNG images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if path:
            self._path_b = path
            self._lbl_path_b.config(text=os.path.basename(path))
            self._load_thumbnail(path, self._lbl_thumb_b)
            self._update_compare_btn()

    def _update_compare_btn(self):
        if self._path_a and self._path_b:
            self._btn_compare.config(state=tk.NORMAL)

    # -----------------------------------------------------------------------
    # Compare pipeline
    # -----------------------------------------------------------------------

    def _run_compare(self):
        self._btn_compare.config(state=tk.DISABLED)
        self._btn_save.config(state=tk.DISABLED)
        self.file_menu.entryconfig("Save Result...", state=tk.DISABLED)
        self._set_status("Loading images...")
        self._progress.start(10)

        thread = threading.Thread(target=self._compare_worker, daemon=True)
        thread.start()

    def _load_cv2_with_exif(self, path: str):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            return None

    def _compare_worker(self):
        try:
            img_a = self._load_cv2_with_exif(self._path_a)
            img_b = self._load_cv2_with_exif(self._path_b)

            if img_a is None: raise ValueError(f"Cannot read Image A: {self._path_a} (may be corrupted)")
            if img_b is None: raise ValueError(f"Cannot read Image B: {self._path_b} (may be corrupted)")

            self._set_status("Aligning images...")
            warped_b, confidence, method, _ = align_images(img_a, img_b, self.cfg)
            low_conf = (confidence == "low")

            self._set_status("Detecting differences...")
            contours, diff_mask, ssim_score = detect_differences(img_a, warped_b, self.cfg)

            from classify import classify_difference
            labels = []
            for c in contours:
                labels.append(classify_difference(c, img_a, warped_b))

            total_area = img_a.shape[0] * img_a.shape[1]
            flagged_area = sum(cv2.contourArea(c) for c in contours)
            pct = (flagged_area / total_area) * 100

            self._set_status("Building composite...")
            composite = build_composite(img_a, warped_b, contours, self.cfg, low_confidence=low_conf, pct_area=pct, labels=labels)

            self._composite = composite
            self.after(0, self._show_composite, composite, len(contours), ssim_score, confidence, method, labels, pct, diff_mask)

        except Exception as exc:
            logger.exception("Pipeline error")
            self.after(0, self._on_error, str(exc))

    def _show_composite(self, composite: np.ndarray, n_diff: int, ssim_score: float, confidence: str, method: str, labels: list, pct: float, diff_mask: np.ndarray):
        self._progress.stop()

        # Convert to RGB PIL Image
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        self._pil_img = Image.fromarray(rgb)
        
        self._fit_to_screen()

        self._btn_compare.config(state=tk.NORMAL)
        self._btn_save.config(state=tk.NORMAL)
        self.file_menu.entryconfig("Save Result...", state=tk.NORMAL)

        status = f"Done — {n_diff} difference(s) found | SSIM={ssim_score:.4f} | Alignment: {method} [{confidence}]"
        self._set_status(status)

        # Update Summary Panel
        from collections import Counter
        counts = Counter(labels)
        breakdown = ", ".join([f"{count} {label}" for label, count in counts.items()])
        if not breakdown:
            breakdown = "0"
            
        summary_text = f"Total Detected: {n_diff}\n"
        summary_text += f"Types: {breakdown}\n"
        summary_text += f"Flagged Area: {pct:.2f}%\n"
        summary_text += f"Alignment: {confidence.upper()}"
        
        self._lbl_summary.config(text=summary_text)
        
        # Update Diff Thumbnail
        # Convert binary diff mask to heatmap
        heatmap = cv2.applyColorMap(diff_mask, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        thumb_img = Image.fromarray(heatmap_rgb)
        thumb_img.thumbnail((250, 150))
        tk_thumb = ImageTk.PhotoImage(thumb_img)
        self._lbl_diff_thumb.config(image=tk_thumb)
        self._lbl_diff_thumb.image = tk_thumb

    def _on_error(self, msg: str):
        self._progress.stop()
        self._btn_compare.config(state=tk.NORMAL)
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Pipeline Error", msg)

    # -----------------------------------------------------------------------
    # Canvas Zoom and Pan
    # -----------------------------------------------------------------------

    def _redraw_canvas(self):
        if self._pil_img is None:
            return
            
        w, h = int(self._pil_img.width * self._zoom_factor), int(self._pil_img.height * self._zoom_factor)
        # Fast resize for display panning
        resized = self._pil_img.resize((w, h), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        self._canvas.configure(scrollregion=(0, 0, w, h))

    def _zoom_in(self):
        self._zoom_factor *= 1.25
        self._redraw_canvas()

    def _zoom_out(self):
        self._zoom_factor *= 0.8
        self._redraw_canvas()

    def _fit_to_screen(self):
        if self._pil_img is None: return
        self.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
            
        scale_w = cw / self._pil_img.width
        scale_h = ch / self._pil_img.height
        self._zoom_factor = min(scale_w, scale_h, 1.0) # max 1x initially
        self._redraw_canvas()

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._zoom_in()
        elif event.delta < 0:
            self._zoom_out()

    def _start_pan(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    # -----------------------------------------------------------------------
    # Save result
    # -----------------------------------------------------------------------

    def _save_result(self):
        if self._composite is None: return
        path = filedialog.asksaveasfilename(
            title="Save Composite",
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")]
        )
        if not path: return
        
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jpg":
            cv2.imwrite(path, self._composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, self._composite)
        self._set_status(f"Saved: {path}")

    def _set_status(self, text: str):
        self.after(0, self._status_var.set, text)

