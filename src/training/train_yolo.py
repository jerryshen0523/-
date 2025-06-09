"""
YOLOv8 滑板車偵測模型訓練模組
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

class ScooterYOLOTrainer:
    def __init__(self, data_config_path: str):
        self.data_config_path = Path(data_config_path)
        self.model_save_dir = Path("models")
        self.model_save_dir.mkdir(exist_ok=True)
        
        # 檢查 GPU 可用性
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")
        
    def load_config(self):
        """載入資料配置"""
        if not self.data_config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {self.data_config_path}")
            
        with open(self.data_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print(f"資料配置:")
        print(f"  資料集路徑: {config['path']}")
        print(f"  類別數量: {config['nc']}")
        print(f"  類別名稱: {config['names']}")
        
        return config
        
    def create_model(self, model_size: str = 'n'):
        """
        建立 YOLO 模型
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"載入預訓練模型: {model_name}")
        
        try:
            model = YOLO(model_name)
            return model
        except Exception as e:
            print(f"載入模型失敗: {e}")
            print("請確保已安裝 ultralytics 套件")
            raise
            
    def train_model(self, 
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640,
                   model_size: str = 'n',
                   patience: int = 50,
                   save_period: int = 10):
        """
        訓練模型
        
        Args:
            epochs: 訓練輪數
            batch_size: 批次大小
            img_size: 輸入圖片尺寸
            model_size: 模型大小
            patience: 早停耐心值
            save_period: 儲存周期
        """
        # 載入配置
        config = self.load_config()
        
        # 建立模型
        model = self.create_model(model_size)
        
        # 訓練參數
        train_args = {
            'data': str(self.data_config_path),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'project': str(self.model_save_dir),
            'name': f'scooter_yolov8{model_size}',
            'patience': patience,
            'save_period': save_period,
            'val': True,
            'plots': True,
            'verbose': True,
            'seed': 42,
            # 資料增強參數
            'hsv_h': 0.015,      # 色調增強
            'hsv_s': 0.7,        # 飽和度增強
            'hsv_v': 0.4,        # 明度增強
            'degrees': 0.0,      # 旋轉角度
            'translate': 0.1,    # 平移
            'scale': 0.5,        # 縮放
            'shear': 0.0,        # 剪切
            'perspective': 0.0,  # 透視變換
            'flipud': 0.0,       # 垂直翻轉
            'fliplr': 0.5,       # 水平翻轉
            'mosaic': 1.0,       # 馬賽克增強
            'mixup': 0.0,        # 混合增強
            'copy_paste': 0.0,   # 複製貼上增強
        }
        
        print("開始訓練...")
        print(f"訓練參數: {train_args}")
        
        try:
            # 開始訓練
            results = model.train(**train_args)
            
            print("訓練完成！")
            print(f"最佳模型路徑: {results.save_dir}")
            
            # 儲存最終模型
            final_model_path = self.model_save_dir / f"scooter_yolov8{model_size}_final.pt"
            model.save(str(final_model_path))
            print(f"最終模型已儲存: {final_model_path}")
            
            return results
            
        except Exception as e:
            print(f"訓練過程中發生錯誤: {e}")
            raise
            
    def evaluate_model(self, model_path: str):
        """
        評估模型
        
        Args:
            model_path: 模型檔案路徑
        """
        print(f"評估模型: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # 在驗證集上評估
            results = model.val(
                data=str(self.data_config_path),
                device=self.device,
                plots=True,
                save_json=True
            )
            
            print("評估結果:")
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")
            print(f"  Precision: {results.box.mp:.4f}")
            print(f"  Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"評估過程中發生錯誤: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='訓練 YOLOv8 滑板車偵測模型')
    parser.add_argument('--data', type=str, required=True, help='資料配置檔案路徑')
    parser.add_argument('--epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='輸入圖片尺寸')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--save-period', type=int, default=10, help='儲存周期')
    parser.add_argument('--eval-only', action='store_true', help='僅評估模型')
    parser.add_argument('--model-path', type=str, help='評估模型的路徑')
    
    args = parser.parse_args()
    
    # 檢查資料配置檔案
    if not Path(args.data).exists():
        print(f"錯誤: 資料配置檔案不存在: {args.data}")
        print("請先執行資料預處理: python src/data_preprocessing/prepare_dataset.py")
        return
        
    trainer = ScooterYOLOTrainer(args.data)
    
    if args.eval_only:
        if not args.model_path:
            print("錯誤: 評估模式需要提供 --model-path 參數")
            return
        trainer.evaluate_model(args.model_path)
    else:
        # 訓練模型
        results = trainer.train_model(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            model_size=args.model_size,
            patience=args.patience,
            save_period=args.save_period
        )
        
        # 找到最佳模型並評估
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            print("\n評估最佳模型...")
            trainer.evaluate_model(str(best_model_path))

if __name__ == "__main__":
    main()
