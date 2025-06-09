"""
資料預處理模組
將標註好的資料轉換為 YOLO 格式，並準備訓練/驗證資料集
"""

import os
import json
import shutil
from pathlib import Path
import yaml
from PIL import Image
import random
from typing import List, Tuple

class DatasetPreprocessor:
    def __init__(self):
        self.base_dir = Path(".")
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.labels_dir = self.data_dir / "labels"
        
        # YOLO 資料集目錄
        self.yolo_dir = self.data_dir / "yolo_dataset"
        self.train_images_dir = self.yolo_dir / "images" / "train"
        self.val_images_dir = self.yolo_dir / "images" / "val"
        self.train_labels_dir = self.yolo_dir / "labels" / "train"
        self.val_labels_dir = self.yolo_dir / "labels" / "val"
        
        # 類別定義
        self.classes = ["scooter"]
        
    def create_directories(self):
        """建立 YOLO 資料集目錄結構"""
        directories = [
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def convert_annotation_to_yolo(self, annotation_data: dict, image_width: int, image_height: int) -> List[str]:
        """
        將標註資料轉換為 YOLO 格式
        
        Args:
            annotation_data: 標註資料
            image_width: 圖片寬度
            image_height: 圖片高度
            
        Returns:
            YOLO 格式的標註行列表
        """
        yolo_lines = []
        
        for annotation in annotation_data.get('annotations', []):
            # 獲取邊界框座標
            x = annotation['x']
            y = annotation['y']
            width = annotation['width']
            height = annotation['height']
            
            # 轉換為 YOLO 格式 (中心點座標 + 相對寬高)
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            rel_width = width / image_width
            rel_height = height / image_height
            
            # YOLO 格式: class_id center_x center_y width height
            class_id = 0  # scooter 類別 ID
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {rel_width:.6f} {rel_height:.6f}"
            yolo_lines.append(yolo_line)
            
        return yolo_lines
        
    def process_annotations(self):
        """處理所有標註檔案"""
        if not self.annotations_dir.exists():
            print(f"標註目錄不存在: {self.annotations_dir}")
            return []
            
        processed_files = []
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        print(f"找到 {len(annotation_files)} 個標註檔案")
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    
                # 獲取對應的圖片檔案
                image_path = Path(annotation_data['image_path'])
                if not image_path.exists():
                    # 嘗試在正樣本目錄中找到圖片
                    image_name = image_path.name
                    possible_paths = [
                        self.images_dir / "positive" / image_name,
                        self.images_dir / image_name
                    ]
                    
                    image_path = None
                    for possible_path in possible_paths:
                        if possible_path.exists():
                            image_path = possible_path
                            break
                            
                    if image_path is None:
                        print(f"找不到圖片: {annotation_data['image_path']}")
                        continue
                        
                # 獲取圖片尺寸
                try:
                    with Image.open(image_path) as img:
                        image_width, image_height = img.size
                except Exception as e:
                    print(f"無法讀取圖片 {image_path}: {e}")
                    continue
                    
                # 轉換標註格式
                yolo_lines = self.convert_annotation_to_yolo(
                    annotation_data, image_width, image_height
                )
                
                if yolo_lines:  # 只處理有標註的圖片
                    processed_files.append({
                        'image_path': image_path,
                        'yolo_lines': yolo_lines,
                        'stem': image_path.stem
                    })
                    
            except Exception as e:
                print(f"處理標註檔案 {annotation_file} 時發生錯誤: {e}")
                
        print(f"成功處理 {len(processed_files)} 個檔案")
        return processed_files
        
    def split_dataset(self, processed_files: List[dict], train_ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
        """
        分割資料集為訓練集和驗證集
        
        Args:
            processed_files: 處理過的檔案列表
            train_ratio: 訓練集比例
            
        Returns:
            (train_files, val_files)
        """
        # 隨機打亂
        random.shuffle(processed_files)
        
        # 計算分割點
        split_index = int(len(processed_files) * train_ratio)
        
        train_files = processed_files[:split_index]
        val_files = processed_files[split_index:]
        
        print(f"資料集分割: 訓練集 {len(train_files)} 張，驗證集 {len(val_files)} 張")
        
        return train_files, val_files
        
    def copy_files_and_create_labels(self, files: List[dict], images_dir: Path, labels_dir: Path):
        """
        複製圖片檔案並建立標籤檔案
        
        Args:
            files: 檔案列表
            images_dir: 目標圖片目錄
            labels_dir: 目標標籤目錄
        """
        for file_info in files:
            image_path = file_info['image_path']
            yolo_lines = file_info['yolo_lines']
            stem = file_info['stem']
            
            # 複製圖片
            target_image_path = images_dir / f"{stem}{image_path.suffix}"
            shutil.copy2(image_path, target_image_path)
            
            # 建立標籤檔案
            target_label_path = labels_dir / f"{stem}.txt"
            with open(target_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
    def create_yaml_config(self):
        """建立 YOLO 訓練配置檔案"""
        config = {
            'path': str(self.yolo_dir.absolute()),  # 資料集根目錄
            'train': 'images/train',  # 訓練圖片相對路徑
            'val': 'images/val',      # 驗證圖片相對路徑
            'nc': len(self.classes),  # 類別數量
            'names': self.classes     # 類別名稱
        }
        
        config_path = self.yolo_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        print(f"已建立配置檔案: {config_path}")
        return config_path
        
    def add_negative_samples(self, negative_ratio: float = 0.3):
        """
        添加負樣本到訓練集
        
        Args:
            negative_ratio: 負樣本在總樣本中的比例
        """
        negative_dir = self.images_dir / "negative"
        if not negative_dir.exists():
            print("負樣本目錄不存在，跳過添加負樣本")
            return
            
        negative_images = list(negative_dir.glob("*.jpg")) + list(negative_dir.glob("*.png"))
        if not negative_images:
            print("負樣本目錄中沒有圖片")
            return
            
        # 計算需要的負樣本數量
        total_positive = len(list(self.train_images_dir.glob("*")))
        target_negative = int(total_positive * negative_ratio / (1 - negative_ratio))
        target_negative = min(target_negative, len(negative_images))
        
        # 隨機選擇負樣本
        selected_negative = random.sample(negative_images, target_negative)
        
        print(f"添加 {len(selected_negative)} 張負樣本到訓練集")
        
        for negative_image in selected_negative:
            # 複製圖片到訓練目錄
            target_path = self.train_images_dir / f"neg_{negative_image.name}"
            shutil.copy2(negative_image, target_path)
            
            # 建立空的標籤檔案（負樣本沒有物件）
            label_path = self.train_labels_dir / f"neg_{negative_image.stem}.txt"
            label_path.touch()  # 建立空檔案
            
    def run_preprocessing(self, train_ratio: float = 0.8, add_negatives: bool = True):
        """
        執行完整的資料預處理流程
        
        Args:
            train_ratio: 訓練集比例
            add_negatives: 是否添加負樣本
        """
        print("開始資料預處理...")
        
        # 建立目錄結構
        self.create_directories()
        
        # 處理標註
        processed_files = self.process_annotations()
        if not processed_files:
            print("沒有找到有效的標註檔案")
            return
            
        # 分割資料集
        train_files, val_files = self.split_dataset(processed_files, train_ratio)
        
        # 複製檔案並建立標籤
        print("建立訓練集...")
        self.copy_files_and_create_labels(train_files, self.train_images_dir, self.train_labels_dir)
        
        print("建立驗證集...")
        self.copy_files_and_create_labels(val_files, self.val_images_dir, self.val_labels_dir)
        
        # 添加負樣本
        if add_negatives:
            self.add_negative_samples()
            
        # 建立配置檔案
        config_path = self.create_yaml_config()
        
        print("資料預處理完成！")
        print(f"資料集目錄: {self.yolo_dir}")
        print(f"配置檔案: {config_path}")
        
        return config_path

def main():
    preprocessor = DatasetPreprocessor()
    config_path = preprocessor.run_preprocessing()
    
    if config_path:
        print(f"\n可以使用以下命令開始訓練:")
        print(f"python src/training/train_yolo.py --data {config_path}")

if __name__ == "__main__":
    main()
