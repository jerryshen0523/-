"""
Real-time Scooter Detection Module
Using computer camera for real-time scooter detection
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
import os

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RealTimeScooterDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, camera_id: int = 0):
        """
        Initialize real-time scooter detector
        
        Args:
            model_path: Path to model file
            conf_threshold: Confidence threshold
            camera_id: Camera ID (usually 0)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.camera_id = camera_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.load_model()
        
        # Initialize camera
        self.init_camera()
        
        # Statistics variables
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
    def load_model(self):
        """Load YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.model = YOLO(str(self.model_path))
            print(f"✅ Model loaded: {self.model_path}")
            print(f"🖥️  Device: {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
            
    def init_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual settings
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Camera initialized: {width}x{height} @ {fps}fps")
        
    def detect_frame(self, frame):
        """
        Detect objects in single frame
        
        Args:
            frame: Input frame
            
        Returns:
            list: Detection results
        """
        # Run detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
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
                
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection results on frame
        
        Args:
            frame: Original frame
            detections: Detection results
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Scooter {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame
    
    def draw_info(self, frame, detections):
        """
        Draw information panel on frame
        
        Args:
            frame: Frame
            detections: Detection results
        """
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
        
        # Draw info panel background
        info_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display information
        info_texts = [
            f"FPS: {self.fps:.1f}",
            f"Scooters detected: {len(detections)}",
            f"Confidence threshold: {self.conf_threshold:.2f}",
            "Press 'q' to quit, 'r' to record"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 20 + i * 20
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self, save_video: bool = False):
        """
        Run real-time detection
        
        Args:
            save_video: Whether to save video
        """
        print("🚀 Starting real-time detection...")
        print("💡 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to start/stop recording")
        print("   - Press '+' to increase confidence threshold")
        print("   - Press '-' to decrease confidence threshold")
        print("   - Press 's' to take screenshot")
        
        video_writer = None
        recording = False
        screenshot_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Cannot read camera frame")
                    break
                
                # Run detection
                detections = self.detect_frame(frame)
                
                # Draw detection results
                annotated_frame = self.draw_detections(frame, detections)
                
                # Draw info panel
                self.draw_info(annotated_frame, detections)
                
                # If recording, write to video
                if recording and video_writer is not None:
                    video_writer.write(annotated_frame)
                    # Show recording indicator
                    cv2.circle(annotated_frame, (annotated_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                
                # Display frame
                cv2.imshow('Real-time Scooter Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Exiting...")
                    break
                elif key == ord('r'):
                    if not recording:
                        # Start recording
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_path = f"results/recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            video_path, fourcc, 20.0, 
                            (annotated_frame.shape[1], annotated_frame.shape[0])
                        )
                        recording = True
                        print(f"🔴 Recording started: {video_path}")
                    else:
                        # Stop recording
                        recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        print("⏹️  Recording stopped")
                
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                    print(f"📈 Confidence threshold: {self.conf_threshold:.2f}")
                
                elif key == ord('-'):
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"📉 Confidence threshold: {self.conf_threshold:.2f}")
                
                elif key == ord('s'):
                    # Take screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"results/screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    screenshot_count += 1
                    print(f"📸 Screenshot saved: {screenshot_path}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupt received, shutting down...")
        
        finally:
            # Clean up resources
            if video_writer:
                video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Resources released")


def main():
    parser = argparse.ArgumentParser(description='Real-time Scooter Detection System')
    parser.add_argument('--model', type=str, 
                       default='models/scooter_yolov8n4/weights/best.pt',
                       help='Path to model file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save-video', action='store_true',
                       help='Automatically save video')
    
    args = parser.parse_args()
    
    # Check model file
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("💡 Please check if the model path is correct")
        return
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Initialize detector
        detector = RealTimeScooterDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            camera_id=args.camera
        )
        
        # Run real-time detection
        detector.run_detection(save_video=args.save_video)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
