"""
滑板車圖片標註工具
使用 LabelImg 風格的介面來標註滑板車位置
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np

class ScooterAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("滑板車標註工具")
        self.root.geometry("1200x800")
        
        # 變數初始化
        self.current_image = None
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        self.annotations = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        
        # 圖片目錄
        self.image_dir = Path("data/images/positive")
        self.output_dir = Path("data/annotations")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        self.load_images()
        
    def setup_ui(self):
        """設置使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側控制面板
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 檔案操作
        ttk.Label(control_frame, text="檔案操作", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        ttk.Button(control_frame, text="選擇圖片目錄", command=self.select_directory).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="載入圖片", command=self.load_images).pack(fill=tk.X, pady=2)
        
        # 圖片導航
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="圖片導航", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="上一張", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav_frame, text="下一張", command=self.next_image).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        self.image_info_label = ttk.Label(control_frame, text="0/0")
        self.image_info_label.pack(pady=5)
        
        # 標註操作
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="標註操作", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        ttk.Button(control_frame, text="清除標註", command=self.clear_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="儲存標註", command=self.save_annotation).pack(fill=tk.X, pady=2)
        
        # 標註列表
        ttk.Label(control_frame, text="當前標註", font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        # 建立 Treeview 來顯示標註
        self.annotation_tree = ttk.Treeview(control_frame, columns=('x', 'y', 'w', 'h'), show='headings', height=6)
        self.annotation_tree.heading('#1', text='X')
        self.annotation_tree.heading('#2', text='Y') 
        self.annotation_tree.heading('#3', text='W')
        self.annotation_tree.heading('#4', text='H')
        self.annotation_tree.column('#1', width=40)
        self.annotation_tree.column('#2', width=40)
        self.annotation_tree.column('#3', width=40)
        self.annotation_tree.column('#4', width=40)
        self.annotation_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 刪除選中的標註
        ttk.Button(control_frame, text="刪除選中", command=self.delete_selected_annotation).pack(fill=tk.X, pady=2)
        
        # 右側圖片顯示區域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 建立畫布
        self.canvas = tk.Canvas(image_frame, bg='white', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 綁定滑鼠事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
        # 狀態列
        self.status_var = tk.StringVar()
        self.status_var.set("請選擇圖片目錄並載入圖片")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_directory(self):
        """選擇圖片目錄"""
        directory = filedialog.askdirectory(initialdir="data/images/positive")
        if directory:
            self.image_dir = Path(directory)
            self.status_var.set(f"已選擇目錄: {directory}")
            
    def load_images(self):
        """載入圖片列表"""
        if not self.image_dir.exists():
            messagebox.showerror("錯誤", f"目錄不存在: {self.image_dir}")
            return
            
        # 支援的圖片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_list = [f for f in self.image_dir.iterdir() 
                          if f.suffix.lower() in image_extensions]
        
        if not self.image_list:
            messagebox.showwarning("警告", "目錄中沒有找到圖片檔案")
            return
            
        self.current_index = 0
        self.show_image()
        self.update_image_info()
        self.status_var.set(f"已載入 {len(self.image_list)} 張圖片")
        
    def show_image(self):
        """顯示當前圖片"""
        if not self.image_list:
            return
            
        self.current_image_path = self.image_list[self.current_index]
        
        # 載入圖片
        image = Image.open(self.current_image_path)
        
        # 計算縮放比例以適應畫布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 畫布尚未初始化，使用預設大小
            canvas_width = 800
            canvas_height = 600
            
        img_width, img_height = image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale = min(scale_x, scale_y, 1.0)  # 不要放大圖片
        
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image)
        
        # 清除畫布並顯示圖片
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.current_image, anchor=tk.CENTER)
        
        # 載入已有的標註
        self.load_existing_annotation()
        
    def load_existing_annotation(self):
        """載入已有的標註"""
        if not self.current_image_path:
            return
            
        annotation_file = self.output_dir / f"{self.current_image_path.stem}.json"
        self.annotations = []
        
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', [])
            except Exception as e:
                print(f"載入標註失敗: {e}")
                
        self.update_annotation_display()
        
    def start_drawing(self, event):
        """開始繪製矩形"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        
    def draw_rectangle(self, event):
        """繪製矩形"""
        if not self.drawing:
            return
            
        # 刪除之前的臨時矩形
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            
        # 繪製新的矩形
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
        
    def end_drawing(self, event):
        """結束繪製矩形"""
        if not self.drawing:
            return
            
        self.drawing = False
        
        # 計算矩形座標
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        
        # 檢查矩形大小
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            return
            
        # 轉換為原始圖片座標
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 計算圖片在畫布中的位置
        img_x = (canvas_width - self.current_image.width()) // 2
        img_y = (canvas_height - self.current_image.height()) // 2
        
        # 轉換座標
        orig_x1 = max(0, (x1 - img_x) / self.scale)
        orig_y1 = max(0, (y1 - img_y) / self.scale)
        orig_x2 = min(self.current_image.width() / self.scale, (x2 - img_x) / self.scale)
        orig_y2 = min(self.current_image.height() / self.scale, (y2 - img_y) / self.scale)
        
        # 添加到標註列表
        annotation = {
            'x': int(orig_x1),
            'y': int(orig_y1),
            'width': int(orig_x2 - orig_x1),
            'height': int(orig_y2 - orig_y1),
            'class': 'scooter'
        }
        
        self.annotations.append(annotation)
        self.update_annotation_display()
        self.rect_id = None
        
    def update_annotation_display(self):
        """更新標註顯示"""
        # 清除 Treeview
        for item in self.annotation_tree.get_children():
            self.annotation_tree.delete(item)
            
        # 清除畫布上的標註矩形
        self.canvas.delete("annotation")
        
        # 顯示所有標註
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_x = (canvas_width - self.current_image.width()) // 2
        img_y = (canvas_height - self.current_image.height()) // 2
        
        for i, ann in enumerate(self.annotations):
            # 添加到 Treeview
            self.annotation_tree.insert('', 'end', values=(
                ann['x'], ann['y'], ann['width'], ann['height']
            ))
            
            # 在畫布上繪製矩形
            x1 = img_x + ann['x'] * self.scale
            y1 = img_y + ann['y'] * self.scale
            x2 = x1 + ann['width'] * self.scale
            y2 = y1 + ann['height'] * self.scale
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='green', width=2, tags="annotation"
            )
            
    def clear_annotations(self):
        """清除所有標註"""
        self.annotations = []
        self.update_annotation_display()
        
    def delete_selected_annotation(self):
        """刪除選中的標註"""
        selection = self.annotation_tree.selection()
        if not selection:
            return
            
        # 獲取選中項目的索引
        item = selection[0]
        index = self.annotation_tree.index(item)
        
        # 刪除標註
        if 0 <= index < len(self.annotations):
            del self.annotations[index]
            self.update_annotation_display()
            
    def save_annotation(self):
        """儲存當前圖片的標註"""
        if not self.current_image_path:
            return
            
        annotation_file = self.output_dir / f"{self.current_image_path.stem}.json"
        
        # 準備儲存的資料
        data = {
            'image_path': str(self.current_image_path),
            'image_width': Image.open(self.current_image_path).width,
            'image_height': Image.open(self.current_image_path).height,
            'annotations': self.annotations
        }
        
        try:
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.status_var.set(f"已儲存標註: {annotation_file.name}")
        except Exception as e:
            messagebox.showerror("錯誤", f"儲存標註失敗: {e}")
            
    def prev_image(self):
        """上一張圖片"""
        if not self.image_list:
            return
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.show_image()
        self.update_image_info()
        
    def next_image(self):
        """下一張圖片"""
        if not self.image_list:
            return
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.show_image()
        self.update_image_info()
        
    def update_image_info(self):
        """更新圖片資訊"""
        if self.image_list:
            self.image_info_label.config(
                text=f"{self.current_index + 1}/{len(self.image_list)}"
            )
        else:
            self.image_info_label.config(text="0/0")

def main():
    root = tk.Tk()
    app = ScooterAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
