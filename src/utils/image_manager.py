"""
圖片管理工具
用於批次處理和組織滑板車偵測專案的圖片
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np

class ImageManager:
    def __init__(self):
        self.project_root = Path(".")
        self.positive_dir = self.project_root / "data" / "images" / "positive"
        self.negative_dir = self.project_root / "data" / "images" / "negative"
        
        # 確保目錄存在
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # 支援的圖片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def copy_images_to_positive(self, source_paths):
        """
        將圖片複製到正樣本目錄
        
        Args:
            source_paths: 來源圖片路徑列表
        """
        copied_count = 0
        failed_count = 0
        
        for source_path in source_paths:
            source_path = Path(source_path)
            
            if not source_path.exists():
                print(f"檔案不存在: {source_path}")
                failed_count += 1
                continue
                
            if source_path.suffix.lower() not in self.supported_formats:
                print(f"不支援的格式: {source_path}")
                failed_count += 1
                continue
                
            try:
                # 生成唯一的檔案名稱
                target_name = self.generate_unique_name(source_path.name, self.positive_dir)
                target_path = self.positive_dir / target_name
                
                # 複製檔案
                shutil.copy2(source_path, target_path)
                print(f"已複製: {source_path.name} -> {target_name}")
                copied_count += 1
                
            except Exception as e:
                print(f"複製失敗 {source_path}: {e}")
                failed_count += 1
                
        print(f"\n複製完成: 成功 {copied_count} 張，失敗 {failed_count} 張")
        return copied_count, failed_count
        
    def copy_images_to_negative(self, source_paths):
        """
        將圖片複製到負樣本目錄
        
        Args:
            source_paths: 來源圖片路徑列表
        """
        copied_count = 0
        failed_count = 0
        
        for source_path in source_paths:
            source_path = Path(source_path)
            
            if not source_path.exists():
                print(f"檔案不存在: {source_path}")
                failed_count += 1
                continue
                
            if source_path.suffix.lower() not in self.supported_formats:
                print(f"不支援的格式: {source_path}")
                failed_count += 1
                continue
                
            try:
                # 生成唯一的檔案名稱
                target_name = self.generate_unique_name(source_path.name, self.negative_dir)
                target_path = self.negative_dir / target_name
                
                # 複製檔案
                shutil.copy2(source_path, target_path)
                print(f"已複製: {source_path.name} -> {target_name}")
                copied_count += 1
                
            except Exception as e:
                print(f"複製失敗 {source_path}: {e}")
                failed_count += 1
                
        print(f"\n複製完成: 成功 {copied_count} 張，失敗 {failed_count} 張")
        return copied_count, failed_count
        
    def generate_unique_name(self, original_name, target_dir):
        """
        生成唯一的檔案名稱
        
        Args:
            original_name: 原始檔案名稱
            target_dir: 目標目錄
            
        Returns:
            唯一的檔案名稱
        """
        name_stem = Path(original_name).stem
        name_suffix = Path(original_name).suffix
        
        counter = 1
        new_name = original_name
        
        while (target_dir / new_name).exists():
            new_name = f"{name_stem}_{counter:03d}{name_suffix}"
            counter += 1
            
        return new_name
        
    def scan_directory(self, directory_path):
        """
        掃描目錄中的所有圖片
        
        Args:
            directory_path: 目錄路徑
            
        Returns:
            圖片檔案路徑列表
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            return []
            
        image_files = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
                
        return sorted(image_files)
        
    def validate_images(self, image_paths):
        """
        驗證圖片檔案是否有效
        
        Args:
            image_paths: 圖片路徑列表
            
        Returns:
            (valid_images, invalid_images)
        """
        valid_images = []
        invalid_images = []
        
        for image_path in image_paths:
            try:
                # 嘗試載入圖片
                with Image.open(image_path) as img:
                    # 檢查圖片尺寸
                    width, height = img.size
                    if width < 32 or height < 32:
                        invalid_images.append((image_path, "圖片太小"))
                        continue
                        
                    # 檢查圖片格式
                    if img.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'webp']:
                        invalid_images.append((image_path, "不支援的格式"))
                        continue
                        
                    valid_images.append(image_path)
                    
            except Exception as e:
                invalid_images.append((image_path, str(e)))
                
        return valid_images, invalid_images
        
    def resize_images(self, image_paths, max_size=(1024, 1024), quality=95):
        """
        調整圖片尺寸以節省空間
        
        Args:
            image_paths: 圖片路徑列表
            max_size: 最大尺寸 (寬, 高)
            quality: JPEG 品質 (1-100)
        """
        processed_count = 0
        
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    # 檢查是否需要調整尺寸
                    if img.size[0] <= max_size[0] and img.size[1] <= max_size[1]:
                        continue
                        
                    # 計算新尺寸
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # 儲存調整後的圖片
                    if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                        img.save(image_path, format='JPEG', quality=quality, optimize=True)
                    else:
                        img.save(image_path, optimize=True)
                        
                    processed_count += 1
                    print(f"已調整尺寸: {image_path.name}")
                    
            except Exception as e:
                print(f"調整尺寸失敗 {image_path}: {e}")
                
        print(f"調整尺寸完成，處理了 {processed_count} 張圖片")
        
    def get_dataset_statistics(self):
        """
        獲取資料集統計資訊
        
        Returns:
            統計字典
        """
        positive_images = self.scan_directory(self.positive_dir)
        negative_images = self.scan_directory(self.negative_dir)
        
        stats = {
            'positive_count': len(positive_images),
            'negative_count': len(negative_images),
            'total_count': len(positive_images) + len(negative_images),
            'positive_files': positive_images,
            'negative_files': negative_images
        }
        
        return stats

class ImageManagerGUI:
    def __init__(self):
        self.manager = ImageManager()
        self.root = tk.Tk()
        self.root.title("滑板車圖片管理工具")
        self.root.geometry("800x600")
        
        self.setup_ui()
        
    def setup_ui(self):
        """設置使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 標題
        title_label = ttk.Label(main_frame, text="滑板車圖片管理工具", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 統計資訊框架
        stats_frame = ttk.LabelFrame(main_frame, text="資料集統計", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="正在載入統計資訊...")
        self.stats_label.pack()
        
        # 操作按鈕框架
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行按鈕
        row1_frame = ttk.Frame(buttons_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row1_frame, text="添加正樣本圖片", command=self.add_positive_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1_frame, text="添加負樣本圖片", command=self.add_negative_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1_frame, text="從目錄添加正樣本", command=self.add_positive_from_folder).pack(side=tk.LEFT)
        
        # 第二行按鈕
        row2_frame = ttk.Frame(buttons_frame)
        row2_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row2_frame, text="從目錄添加負樣本", command=self.add_negative_from_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2_frame, text="驗證圖片檔案", command=self.validate_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2_frame, text="調整圖片尺寸", command=self.resize_images).pack(side=tk.LEFT)
        
        # 第三行按鈕
        row3_frame = ttk.Frame(buttons_frame)
        row3_frame.pack(fill=tk.X)
        
        ttk.Button(row3_frame, text="重新整理統計", command=self.update_statistics).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row3_frame, text="開啟標註工具", command=self.open_annotation_tool).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row3_frame, text="開啟專案目錄", command=self.open_project_folder).pack(side=tk.LEFT)
        
        # 日誌區域
        log_frame = ttk.LabelFrame(main_frame, text="操作日誌", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 建立文字區域和捲動條
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化統計
        self.update_statistics()
        
    def log_message(self, message):
        """添加日誌訊息"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def add_positive_images(self):
        """添加正樣本圖片"""
        files = filedialog.askopenfilenames(
            title="選擇包含滑板車的圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("所有檔案", "*.*")
            ]
        )
        
        if files:
            self.log_message(f"開始添加 {len(files)} 張正樣本圖片...")
            copied, failed = self.manager.copy_images_to_positive(files)
            self.log_message(f"正樣本添加完成: 成功 {copied} 張，失敗 {failed} 張")
            self.update_statistics()
            
    def add_negative_images(self):
        """添加負樣本圖片"""
        files = filedialog.askopenfilenames(
            title="選擇不包含滑板車的圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("所有檔案", "*.*")
            ]
        )
        
        if files:
            self.log_message(f"開始添加 {len(files)} 張負樣本圖片...")
            copied, failed = self.manager.copy_images_to_negative(files)
            self.log_message(f"負樣本添加完成: 成功 {copied} 張，失敗 {failed} 張")
            self.update_statistics()
            
    def add_positive_from_folder(self):
        """從目錄添加正樣本"""
        folder = filedialog.askdirectory(title="選擇包含滑板車圖片的目錄")
        
        if folder:
            self.log_message(f"掃描目錄: {folder}")
            images = self.manager.scan_directory(folder)
            
            if images:
                self.log_message(f"找到 {len(images)} 張圖片，開始複製...")
                copied, failed = self.manager.copy_images_to_positive(images)
                self.log_message(f"正樣本添加完成: 成功 {copied} 張，失敗 {failed} 張")
                self.update_statistics()
            else:
                self.log_message("目錄中沒有找到圖片檔案")
                
    def add_negative_from_folder(self):
        """從目錄添加負樣本"""
        folder = filedialog.askdirectory(title="選擇不包含滑板車圖片的目錄")
        
        if folder:
            self.log_message(f"掃描目錄: {folder}")
            images = self.manager.scan_directory(folder)
            
            if images:
                self.log_message(f"找到 {len(images)} 張圖片，開始複製...")
                copied, failed = self.manager.copy_images_to_negative(images)
                self.log_message(f"負樣本添加完成: 成功 {copied} 張，失敗 {failed} 張")
                self.update_statistics()
            else:
                self.log_message("目錄中沒有找到圖片檔案")
                
    def validate_images(self):
        """驗證圖片檔案"""
        self.log_message("開始驗證圖片檔案...")
        
        # 驗證正樣本
        positive_images = self.manager.scan_directory(self.manager.positive_dir)
        if positive_images:
            valid_pos, invalid_pos = self.manager.validate_images(positive_images)
            self.log_message(f"正樣本驗證: {len(valid_pos)} 張有效，{len(invalid_pos)} 張無效")
            
            for img_path, error in invalid_pos:
                self.log_message(f"  無效: {img_path.name} - {error}")
                
        # 驗證負樣本
        negative_images = self.manager.scan_directory(self.manager.negative_dir)
        if negative_images:
            valid_neg, invalid_neg = self.manager.validate_images(negative_images)
            self.log_message(f"負樣本驗證: {len(valid_neg)} 張有效，{len(invalid_neg)} 張無效")
            
            for img_path, error in invalid_neg:
                self.log_message(f"  無效: {img_path.name} - {error}")
                
    def resize_images(self):
        """調整圖片尺寸"""
        response = messagebox.askyesno(
            "確認調整尺寸",
            "這將調整所有大於 1024x1024 的圖片尺寸，是否繼續？\n"
            "注意：這個操作會覆蓋原始檔案。"
        )
        
        if response:
            self.log_message("開始調整圖片尺寸...")
            
            # 處理正樣本
            positive_images = self.manager.scan_directory(self.manager.positive_dir)
            if positive_images:
                self.log_message("調整正樣本圖片尺寸...")
                self.manager.resize_images(positive_images)
                
            # 處理負樣本
            negative_images = self.manager.scan_directory(self.manager.negative_dir)
            if negative_images:
                self.log_message("調整負樣本圖片尺寸...")
                self.manager.resize_images(negative_images)
                
            self.log_message("圖片尺寸調整完成")
            
    def update_statistics(self):
        """更新統計資訊"""
        stats = self.manager.get_dataset_statistics()
        
        stats_text = f"正樣本: {stats['positive_count']} 張  |  " \
                    f"負樣本: {stats['negative_count']} 張  |  " \
                    f"總計: {stats['total_count']} 張"
        
        self.stats_label.config(text=stats_text)
        
    def open_annotation_tool(self):
        """開啟標註工具"""
        if self.manager.get_dataset_statistics()['positive_count'] == 0:
            messagebox.showwarning("警告", "請先添加正樣本圖片")
            return
            
        self.log_message("正在啟動標註工具...")
        
        try:
            import subprocess
            subprocess.Popen([
                "python", 
                str(Path("src/annotation_tool/annotate.py").absolute())
            ])
            self.log_message("標註工具已啟動")
        except Exception as e:
            self.log_message(f"啟動標註工具失敗: {e}")
            
    def open_project_folder(self):
        """開啟專案目錄"""
        try:
            import subprocess
            subprocess.Popen(['explorer', str(Path('.').absolute())])
        except Exception as e:
            self.log_message(f"開啟目錄失敗: {e}")
            
    def run(self):
        """執行 GUI"""
        self.root.mainloop()

def main():
    """主函數"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        # 啟動 GUI 模式
        app = ImageManagerGUI()
        app.run()
    else:
        # 命令列模式
        manager = ImageManager()
        stats = manager.get_dataset_statistics()
        
        print("滑板車圖片管理工具")
        print("=" * 30)
        print(f"正樣本: {stats['positive_count']} 張")
        print(f"負樣本: {stats['negative_count']} 張")
        print(f"總計: {stats['total_count']} 張")
        print("\n使用 --gui 參數啟動圖形介面")

if __name__ == "__main__":
    main()
