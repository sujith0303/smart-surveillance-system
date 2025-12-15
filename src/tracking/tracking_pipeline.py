# Complete Multi-Object Tracking System with Behavior Detection
# File: src/tracking/tracking_pipeline.py

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from datetime import datetime
import json
import os

class SurveillanceTracker:
    """
    Multi-object tracking with trajectory analysis
    Tracks objects across frames and detects behaviors
    """
    
    def __init__(self, model_path='yolov8x.pt', conf_threshold=0.5):
        """Initialize tracker with YOLOv8 + built-in tracking"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Track history: {track_id: [positions, timestamps, attributes]}
        self.track_history = defaultdict(lambda: {
            'positions': [],
            'timestamps': [],
            'first_seen': None,
            'last_seen': None,
            'attributes': {},
            'behaviors': []
        })
        
        # Behavior detection parameters
        self.loitering_threshold = 10.0  # seconds
        self.speed_threshold = 50  # pixels per second for "running"
        
    def track_video(self, video_path, output_json='tracking_results.json',
                   frame_skip=2, max_frames=None, visualize=False,
                   output_video=None):
        """
        Track objects throughout video
        
        Args:
            video_path: Input video path
            output_json: Path to save tracking results
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
            visualize: Whether to create annotated video
            output_video: Path for output video (if visualize=True)
        """
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"❌ Error: Video not found at {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
        
        # Setup video writer if visualizing
        out = None
        if visualize and output_video:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_video)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps/frame_skip, 
                                 (width, height))
            print(f"📹 Will save annotated video to: {output_video}")
        
        frame_count = 0
        processed_count = 0
        
        print(f"\n🎬 Starting tracking...")
        print(f"Processing every {frame_skip} frames")
        if max_frames:
            print(f"Max frames to process: {max_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            timestamp = frame_count / fps
            
            # YOLOv8 tracking (includes detection + tracking)
            results = self.model.track(
                frame, 
                conf=self.conf_threshold,
                persist=True,  # Persist tracks between frames
                tracker="bytetrack.yaml",  # Use ByteTrack
                verbose=False
            )
            
            # Process tracking results
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, class_id, conf in zip(boxes, track_ids, 
                                                          classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Get object ROI for attribute extraction
                    roi = frame[y1:y2, x1:x2]
                    
                    # Update track history
                    track = self.track_history[track_id]
                    
                    if track['first_seen'] is None:
                        track['first_seen'] = timestamp
                        track['attributes'] = self._extract_attributes(
                            roi, class_id
                        )
                    
                    track['last_seen'] = timestamp
                    track['positions'].append(center)
                    track['timestamps'].append(timestamp)
                    
                    # Detect behaviors
                    behaviors = self._detect_behaviors(track_id)
                    if behaviors:
                        track['behaviors'].extend(behaviors)
                    
                    # Visualize if enabled
                    if visualize:
                        self._draw_tracking(frame, box, track_id, 
                                          track['attributes'], behaviors)
            
            if visualize and out is not None:
                out.write(frame)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"⏳ Processed {processed_count} frames, "
                      f"{len(self.track_history)} unique tracks")
            
            if max_frames and processed_count >= max_frames:
                print(f"⚠️  Reached max frames limit: {max_frames}")
                break
            
            frame_count += 1
        
        cap.release()
        if out is not None:
            out.release()
            print(f"✅ Annotated video saved: {output_video}")
        
        # Generate tracking report
        report = self._generate_tracking_report()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Tracking complete!")
        print(f"📊 Results:")
        print(f"   Unique tracks: {len(self.track_history)}")
        print(f"   Suspicious tracks: {report['summary']['suspicious_tracks']}")
        print(f"   Total behaviors detected: {sum(len(t['behaviors']) for t in self.track_history.values())}")
        print(f"   Report saved to: {output_json}")
        
        return report
    
    def _extract_attributes(self, roi, class_id):
        """Extract visual attributes from ROI"""
        if roi.size == 0:
            return {'color': 'unknown', 'type': 'unknown'}
        
        color = self._detect_color(roi)
        
        # Map class IDs to types
        class_map = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            24: 'backpack',
            26: 'handbag',
            28: 'suitcase'
        }
        
        return {
            'color': color,
            'type': class_map.get(class_id, 'object'),
            'class_id': int(class_id)
        }
    
    def _detect_color(self, roi):
        """Detect dominant color"""
        try:
            roi_small = cv2.resize(roi, (50, 50))
            hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
            
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            if avg_saturation < 40:
                if avg_value > 200: return 'white'
                elif avg_value < 50: return 'black'
                else: return 'gray'
            
            if avg_hue < 10 or avg_hue > 170: return 'red'
            elif 10 <= avg_hue < 25: return 'orange'
            elif 25 <= avg_hue < 35: return 'yellow'
            elif 35 <= avg_hue < 85: return 'green'
            elif 85 <= avg_hue < 130: return 'blue'
            elif 130 <= avg_hue < 170: return 'purple'
            
            return 'unknown'
        except:
            return 'unknown'
    
    def _detect_behaviors(self, track_id):
        """
        Detect suspicious or unusual behaviors
        Returns list of detected behaviors
        """
        track = self.track_history[track_id]
        behaviors = []
        
        # Need at least 10 positions for behavior analysis
        if len(track['positions']) < 10:
            return behaviors
        
        # 1. Loitering detection (staying in same area)
        recent_positions = track['positions'][-20:]  # Last 20 positions
        if len(recent_positions) >= 10:
            # Calculate movement distance
            total_distance = 0
            for i in range(1, len(recent_positions)):
                p1 = np.array(recent_positions[i-1])
                p2 = np.array(recent_positions[i])
                total_distance += np.linalg.norm(p2 - p1)
            
            avg_movement = total_distance / len(recent_positions)
            
            # If minimal movement for extended time = loitering
            if avg_movement < 5 and len(track['positions']) > 30:
                time_in_area = track['last_seen'] - track['timestamps'][-30]
                if time_in_area > self.loitering_threshold:
                    behaviors.append({
                        'type': 'loitering',
                        'duration': round(time_in_area, 1),
                        'timestamp': track['last_seen']
                    })
        
        # 2. Speed detection (running/fast movement)
        if len(track['positions']) >= 5:
            recent_pos = track['positions'][-5:]
            recent_time = track['timestamps'][-5:]
            
            # Calculate speed
            distance = np.linalg.norm(
                np.array(recent_pos[-1]) - np.array(recent_pos[0])
            )
            time_elapsed = recent_time[-1] - recent_time[0]
            
            if time_elapsed > 0:
                speed = distance / time_elapsed  # pixels per second
                
                if speed > self.speed_threshold:
                    behaviors.append({
                        'type': 'fast_movement',
                        'speed': round(speed, 1),
                        'timestamp': track['last_seen']
                    })
        
        # 3. Direction change (erratic movement)
        if len(track['positions']) >= 15:
            recent = track['positions'][-15:]
            direction_changes = 0
            
            for i in range(2, len(recent)):
                v1 = np.array(recent[i-1]) - np.array(recent[i-2])
                v2 = np.array(recent[i]) - np.array(recent[i-1])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0, 1.0
                    ))
                    
                    if angle > np.pi / 2:  # More than 90 degree change
                        direction_changes += 1
            
            if direction_changes > 5:
                behaviors.append({
                    'type': 'erratic_movement',
                    'direction_changes': direction_changes,
                    'timestamp': track['last_seen']
                })
        
        return behaviors
    
    def _draw_tracking(self, frame, box, track_id, attributes, behaviors):
        """Draw tracking visualization on frame"""
        x1, y1, x2, y2 = map(int, box)
        
        # Color based on behavior
        color = (0, 255, 0)  # Green = normal
        if behaviors:
            color = (0, 0, 255)  # Red = suspicious
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and attributes
        label = f"ID:{track_id} {attributes.get('color', '')} {attributes.get('type', '')}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw behavior alerts
        if behaviors:
            for i, behavior in enumerate(behaviors[-3:]):  # Show last 3
                alert = f"{behavior['type']}"
                cv2.putText(frame, alert, (x1, y2 + 20 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def _generate_tracking_report(self):
        """Generate comprehensive tracking report"""
        report = {
            'summary': {
                'total_tracks': len(self.track_history),
                'suspicious_tracks': 0,
                'total_behaviors': 0
            },
            'tracks': []
        }
        
        for track_id, track in self.track_history.items():
            duration = track['last_seen'] - track['first_seen']
            
            track_report = {
                'track_id': int(track_id),
                'attributes': track['attributes'],
                'first_seen': round(track['first_seen'], 2),
                'last_seen': round(track['last_seen'], 2),
                'duration': round(duration, 2),
                'total_detections': len(track['positions']),
                'behaviors': track['behaviors']
            }
            
            if track['behaviors']:
                report['summary']['suspicious_tracks'] += 1
                report['summary']['total_behaviors'] += len(track['behaviors'])
            
            report['tracks'].append(track_report)
        
        # Sort by first_seen timestamp
        report['tracks'].sort(key=lambda x: x['first_seen'])
        
        return report


# Usage Example
if __name__ == "__main__":
    print("🎬 Smart Surveillance - Object Tracking System")
    print("=" * 60)
    
    tracker = SurveillanceTracker(
        model_path='yolov8x.pt',
        conf_threshold=0.5
    )
    
    report = tracker.track_video(
        video_path='data/videos/test_surveillance.mp4',
        output_json='data/outputs/tracking_report.json',
        frame_skip=2,
        max_frames=500,  # Process first 500 frames (remove for full video)
        visualize=True,
        output_video='data/outputs/tracked_output.mp4'
    )
    
    if report:
        print("\n" + "=" * 60)
        print("📊 TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total unique objects tracked: {report['summary']['total_tracks']}")
        print(f"Suspicious behaviors detected: {report['summary']['suspicious_tracks']}")
        print(f"Total behavior events: {report['summary']['total_behaviors']}")
        
        # Show some example tracks
        if report['tracks']:
            print(f"\n📋 Sample Tracks (first 5):")
            for track in report['tracks'][:5]:
                print(f"\n  Track ID {track['track_id']}:")
                print(f"    Type: {track['attributes'].get('type', 'unknown')}")
                print(f"    Color: {track['attributes'].get('color', 'unknown')}")
                print(f"    Duration: {track['duration']:.1f}s")
                print(f"    Detections: {track['total_detections']}")
                if track['behaviors']:
                    print(f"    ⚠️  Behaviors: {[b['type'] for b in track['behaviors']]}")
        
        print("\n✅ All results saved!")
        print(f"   JSON report: data/outputs/tracking_report.json")
        print(f"   Video output: data/outputs/tracked_output.mp4")