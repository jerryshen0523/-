"""
Batch Scooter Detection Module - English Version
Process multiple images and generate detection report
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
from datetime import datetime
import os

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class BatchScooterDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.1):
        """
        Initialize batch scooter detector
        
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
            print(f"✅ Model loaded: {self.model_path}")
            print(f"🖥️ Device: {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def detect_batch_images(self, image_dir: str, output_dir: str = "results/batch_detection"):
        """
        Detect scooters in batch images
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory for results
            
        Returns:
            dict: Batch detection results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return {}
        
        print(f"📁 Found {len(image_files)} images")
        print(f"💾 Output directory: {output_dir}")
        
        # Batch detection results
        batch_results = {
            'detection_time': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'confidence_threshold': self.conf_threshold,
            'total_images': len(image_files),
            'total_detections': 0,
            'results': []
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔍 Processing [{i}/{len(image_files)}]: {image_path.name}")
            
            try:
                # Run detection
                results = self.model(str(image_path), conf=self.conf_threshold, verbose=False)
                result = results[0]
                
                # Process results
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class': int(cls),
                            'class_name': 'scooter'
                        }
                        detections.append(detection)
                
                # Save detection result
                image_result = {
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'detections_count': len(detections),
                    'detections': detections
                }
                
                batch_results['results'].append(image_result)
                batch_results['total_detections'] += len(detections)
                
                print(f"   ✅ Detected {len(detections)} scooters")
                
                # Save annotated image
                if detections:
                    self.save_annotated_image(image_path, detections, output_dir)
                
            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")
                continue
        
        # Calculate processing time
        total_time = time.time() - start_time
        batch_results['processing_time_seconds'] = total_time
        batch_results['average_time_per_image'] = total_time / len(image_files)
        
        # Save batch results to JSON
        results_file = output_dir / "batch_detection_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self.generate_summary_report(batch_results, output_dir)
        
        print(f"\n📊 Batch Detection Summary:")
        print(f"   Total images processed: {batch_results['total_images']}")
        print(f"   Total scooters detected: {batch_results['total_detections']}")
        print(f"   Average detections per image: {batch_results['total_detections']/batch_results['total_images']:.2f}")
        print(f"   Processing time: {total_time:.2f} seconds")
        print(f"   Average time per image: {total_time/len(image_files):.2f} seconds")
        print(f"   Results saved to: {results_file}")
        
        return batch_results
    
    def save_annotated_image(self, image_path: Path, detections: list, output_dir: Path):
        """
        Save annotated image with detection boxes
        
        Args:
            image_path: Original image path
            detections: Detection results
            output_dir: Output directory
        """
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
        
        ax.set_title(f"Scooter Detection - {image_path.name} ({len(detections)} detected)", fontsize=16)
        
        # Save result
        output_path = output_dir / f"detected_{image_path.name}"
        
        # Set font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def generate_summary_report(self, batch_results: dict, output_dir: Path):
        """
        Generate summary report with statistics and visualizations
        
        Args:
            batch_results: Batch detection results
            output_dir: Output directory
        """
        # Extract statistics
        detection_counts = [result['detections_count'] for result in batch_results['results']]
        confidence_scores = []
        
        for result in batch_results['results']:
            for detection in result['detections']:
                confidence_scores.append(detection['confidence'])
        
        # Create summary plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Detection count distribution
        ax1.hist(detection_counts, bins=max(1, max(detection_counts) if detection_counts else 1), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Scooter Counts per Image')
        ax1.set_xlabel('Number of Scooters Detected')
        ax1.set_ylabel('Number of Images')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence score distribution
        if confidence_scores:
            ax2.hist(confidence_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_title('Distribution of Detection Confidence Scores')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Number of Detections')
            ax2.axvline(self.conf_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {self.conf_threshold}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Detection Confidence Scores')
        
        # Plot 3: Images with/without detections
        images_with_detections = sum(1 for count in detection_counts if count > 0)
        images_without_detections = len(detection_counts) - images_with_detections
        
        labels = ['With Scooters', 'Without Scooters']
        sizes = [images_with_detections, images_without_detections]
        colors = ['lightcoral', 'lightblue']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Images with/without Scooter Detections')
        
        # Plot 4: Statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        stats_data = [
            ['Total Images', batch_results['total_images']],
            ['Total Detections', batch_results['total_detections']],
            ['Images with Scooters', images_with_detections],
            ['Images without Scooters', images_without_detections],
            ['Avg Detections/Image', f"{batch_results['total_detections']/batch_results['total_images']:.2f}"],
            ['Processing Time', f"{batch_results['processing_time_seconds']:.2f}s"],
            ['Avg Time/Image', f"{batch_results['average_time_per_image']:.2f}s"],
            ['Confidence Threshold', batch_results['confidence_threshold']],
        ]
        
        if confidence_scores:
            stats_data.extend([
                ['Min Confidence', f"{min(confidence_scores):.3f}"],
                ['Max Confidence', f"{max(confidence_scores):.3f}"],
                ['Avg Confidence', f"{np.mean(confidence_scores):.3f}"],
            ])
        
        table = ax4.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Detection Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Summary report saved: {output_dir / 'detection_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description='Batch Scooter Detection - English Version')
    parser.add_argument('--model', type=str, 
                       default='models/scooter_yolov8n4/weights/best.pt',
                       help='Path to model file')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, 
                       default='results/batch_detection',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold (default: 0.1)')
    
    args = parser.parse_args()
    
    # Check model file
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Check input directory
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    try:
        # Initialize detector
        detector = BatchScooterDetector(args.model, args.conf)
        
        # Run batch detection
        results = detector.detect_batch_images(args.input_dir, args.output_dir)
        
        print(f"\n✅ Batch detection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
