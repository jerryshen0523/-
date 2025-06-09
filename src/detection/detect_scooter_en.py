"""
Scooter Detection Inference Module - English Version
Using trained YOLO model for scooter detection
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ScooterDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.1):
        """
        Initialize scooter detector
        
        Args:
            model_path: Path to model file
            conf_threshold: Confidence threshold
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.model = YOLO(str(self.model_path))
            print(f"Model loaded: {self.model_path}")
            print(f"Device: {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
            
    def detect_single_image(self, image_path: str, save_result: bool = True):
        """
        Detect scooters in a single image
        
        Args:
            image_path: Path to image
            save_result: Whether to save result
            
        Returns:
            list: Detection results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"Detecting image: {image_path}")
        
        # Run detection
        results = self.model(str(image_path), conf=self.conf_threshold)
        result = results[0]
        
        # Process results
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': 'scooter'
                }
                detections.append(detection)
                
        print(f"Detected {len(detections)} scooters")
        print("Detection details:")
        for i, det in enumerate(detections):
            print(f"  Scooter {i+1}: confidence {det['confidence']:.3f}")
        
        # Save result image
        if save_result:
            self.save_detection_result(image_path, detections)
            
        return detections
    
    def save_detection_result(self, image_path: Path, detections: list):
        """
        Save detection result image
        
        Args:
            image_path: Original image path
            detections: Detection results
        """
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Calculate rectangle parameters
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), width, height,
                                   linewidth=3, edgecolor='red', 
                                   facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"Scooter {conf:.2f}"
            ax.text(x1, y1-10, label,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   fontsize=12, color='white')
        
        ax.set_title(f"Scooter Detection Result - {image_path.name}", fontsize=16)
        
        # Save result
        output_path = results_dir / f"detected_{image_path.name}"
        
        # Set font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Result saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Scooter Detection - English Version')
    parser.add_argument('--model', type=str, 
                       default='models/scooter_yolov8n4/weights/best.pt',
                       help='Path to model file')
    parser.add_argument('--image', type=str,
                       help='Single image path')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold (default: 0.1)')
    
    args = parser.parse_args()
    
    # Check model file
    if not Path(args.model).exists():
        print(f"Error: Model file not found {args.model}")
        return
    
    # Initialize detector
    detector = ScooterDetector(args.model, args.conf)
    
    # Detect single image
    if args.image:
        if Path(args.image).exists():
            detector.detect_single_image(args.image)
        else:
            print(f"Error: Image file not found {args.image}")


if __name__ == "__main__":
    main()
