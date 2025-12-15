# Day 1-2: Basic Object Detection Pipeline
# File: detection_pipeline.py

import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json

class SurveillanceDetector:
    """
    Core object detection system using YOLOv8
    Detects people, vehicles, and objects in video streams
    """
    
    def __init__(self, model_path='yolov8x.pt', conf_threshold=0.5):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Classes we care about for surveillance
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            24: 'backpack',
            26: 'handbag',
            28: 'suitcase'
        }
        
    def detect_frame(self, frame, frame_number, timestamp):
        """
        Detect objects in a single frame
        
        Args:
            frame: Video frame (numpy array)
            frame_number: Frame index
            timestamp: Video timestamp
            
        Returns:
            List of detected events with metadata
        """
        # Run YOLOv8 detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        events = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get detection info
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Only process target classes
                if class_id in self.target_classes:
                    # Extract region of interest for attribute detection
                    x1, y1, x2, y2 = map(int, bbox)
                    roi = frame[y1:y2, x1:x2]
                    
                    # Detect attributes (color, etc)
                    attributes = self._extract_attributes(roi, class_id)
                    
                    # Create event record
                    event = {
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'object_type': self.target_classes[class_id],
                        'confidence': round(confidence, 3),
                        'bbox': [x1, y1, x2, y2],
                        'attributes': attributes,
                        'detection_time': datetime.now().isoformat()
                    }
                    
                    events.append(event)
        
        return events
    
    def _extract_attributes(self, roi, class_id):
        """
        Extract visual attributes from detected object
        
        Args:
            roi: Region of interest (cropped detection)
            class_id: YOLO class ID
            
        Returns:
            Dictionary of attributes
        """
        attributes = {}
        
        if roi.size == 0:
            return attributes
        
        # Detect dominant color
        color = self._detect_color(roi)
        attributes['color'] = color
        
        # Add object-specific attributes
        if class_id == 0:  # Person
            attributes['type'] = 'person'
            # Could add: clothing detection, pose estimation, etc
            
        elif class_id in [2, 3, 5, 7]:  # Vehicles
            attributes['type'] = 'vehicle'
            attributes['vehicle_type'] = self.target_classes[class_id]
            
        elif class_id in [24, 26, 28]:  # Bags
            attributes['type'] = 'bag'
            
        return attributes
    
    def _detect_color(self, roi):
        """
        Detect dominant color in region
        
        Args:
            roi: Image region
            
        Returns:
            Color name string
        """
        # Resize for faster processing
        roi_small = cv2.resize(roi, (50, 50))
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
        
        # Get average hue
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Color classification based on HSV
        if avg_saturation < 40:  # Low saturation = grayscale
            if avg_value > 200:
                return 'white'
            elif avg_value < 50:
                return 'black'
            else:
                return 'gray'
        
        # Color hue ranges
        if avg_hue < 10 or avg_hue > 170:
            return 'red'
        elif 10 <= avg_hue < 25:
            return 'orange'
        elif 25 <= avg_hue < 35:
            return 'yellow'
        elif 35 <= avg_hue < 85:
            return 'green'
        elif 85 <= avg_hue < 130:
            return 'blue'
        elif 130 <= avg_hue < 170:
            return 'purple'
        
        return 'unknown'
    
    def process_video(self, video_path, output_json='detections.json', 
                     frame_skip=5, max_frames=None):
        """
        Process entire video and save detections
        
        Args:
            video_path: Path to video file
            output_json: Path to save detection results
            frame_skip: Process every Nth frame (for speed)
            max_frames: Maximum frames to process (None = all)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_events = []
        frame_count = 0
        processed_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Frame skip: {frame_skip}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Calculate timestamp
            timestamp = frame_count / fps
            
            # Detect objects in frame
            events = self.detect_frame(frame, frame_count, timestamp)
            all_events.extend(events)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} frames, "
                      f"found {len(all_events)} detections")
            
            # Stop if max frames reached
            if max_frames and processed_count >= max_frames:
                break
            
            frame_count += 1
        
        cap.release()
        
        # Save results
        with open(output_json, 'w') as f:
            json.dump(all_events, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"Total frames processed: {processed_count}")
        print(f"Total detections: {len(all_events)}")
        print(f"Results saved to: {output_json}")
        
        return all_events


# Usage Example
if __name__ == "__main__":
    # Initialize detector
    detector = SurveillanceDetector(
        model_path='yolov8x.pt',
        conf_threshold=0.5
    )
    
    # Process a video
    # Replace with your test video path
    video_path = "/Users/bsujithreddy/work/smart-surveillance-system/data/videos/test_surveillance.mp4"
    
    events = detector.process_video(
        video_path=video_path,
        output_json='data/outputs/surveillance_detections.json',
        frame_skip=5,  # Process every 5th frame
        max_frames=1000  # Test with 1000 frames first
    )
    
    # Print summary
    print("\n📊 Detection Summary:")
    object_counts = {}
    for event in events:
        obj_type = event['object_type']
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    for obj_type, count in sorted(object_counts.items()):
        print(f"  {obj_type}: {count}")