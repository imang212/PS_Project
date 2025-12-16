import numpy as np
from datetime import datetime

class SimpleTracker:
    """
    Simple object tracker based on IOU matching. Works directly with detections from Hailo inference engine.
    """
    def __init__(self, max_lost_frames=30, iou_threshold=0.3):
        """
        Args:
            max_lost_frames: Number of frames before object "disappears"
            iou_threshold: Threshold for IOU matching (0-1)
        """
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        # Tracking state
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracker with new detections.
        Args:
            detections: List of detections, each detection is a dict with keys:
                       - 'bbox': [x1, y1, x2, y2] (normalized 0-1)
                       - 'confidence': float (0-1)
                       - 'class_id': int
                       - 'class_name': str
        Returns: List of tracked objects with added 'track_id'
        """
        self.frame_count += 1
        updated_tracks = {}
        assigned_detections = set()
        # Match detections to existing tracks using highest IOU
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = None
            for idx, det in enumerate(detections):
                if idx in assigned_detections:
                    continue
                iou = self._calculate_iou(track.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            if best_iou >= self.iou_threshold and best_det_idx is not None:
                # Update track
                track.update(detections[best_det_idx], self.frame_count)
                updated_tracks[track_id] = track
                assigned_detections.add(best_det_idx)
            else:
                # No matching detection → mark lost
                track.mark_lost(self.frame_count)
                if not track.should_remove(self.frame_count, self.max_lost_frames):
                    updated_tracks[track_id] = track
        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in assigned_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                new_track = Track(track_id, det, self.frame_count)
                updated_tracks[track_id] = new_track
        self.tracks = updated_tracks
        # Return list of active tracks
        return self._get_active_tracks()
    
    @staticmethod
    def _calculate_iou(bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] (normalized 0-1)
        Returns: IOU score (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    
    def _get_active_tracks(self):
        active = []
        for track in self.tracks.values():
            if track.is_active() or track.is_recently_lost(self.frame_count, frames=5):
                obj = {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'age': track.age,
                    'state': track.state,
                    'timestamp': track.timestamp
                }
                active.append(obj)
        return active

class Track:
    STATE_NEW = 'new'
    STATE_TRACKED = 'tracked'
    STATE_LOST = 'lost'

    def __init__(self, track_id, detection, frame_count):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        self.timestamp = detection['timestamp']
        self.state = self.STATE_NEW
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.first_frame = frame_count
        self.last_seen_frame = frame_count

    def update(self, detection, frame_count):
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_frame = frame_count
        self.age += 1
        if self.state == self.STATE_NEW and self.hits >= 3:
            self.state = self.STATE_TRACKED
        elif self.state == self.STATE_LOST:
            self.state = self.STATE_TRACKED

    def mark_lost(self, frame_count):
        self.state = self.STATE_LOST
        self.time_since_update = frame_count - self.last_seen_frame

    def is_active(self):
        return self.state in (self.STATE_NEW, self.STATE_TRACKED)

    def is_recently_lost(self, frame_count, frames=5):
        return self.state == self.STATE_LOST and (frame_count - self.last_seen_frame) <= frames

    def should_remove(self, frame_count, max_lost_frames):
        if self.state == self.STATE_NEW and self.time_since_update > 3:
            return True
        if self.state == self.STATE_LOST:
            frames_lost = frame_count - self.last_seen_frame
            return frames_lost > max_lost_frames
        return False


if __name__ == "__main__":
    # Example usage with HailoInferenceEngine
    tracker = SimpleTracker(max_lost_frames=30, iou_threshold=0.3)
    # Simulate detections from Hailo
    frame_1_detections = [
        {'bbox': [0.1, 0.2, 0.3, 0.5], 'confidence': 0.9, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [0.6, 0.3, 0.8, 0.6], 'confidence': 0.85, 'class_id': 1, 'class_name': 'car'},
    ]
    frame_2_detections = [
        {'bbox': [0.12, 0.22, 0.32, 0.52], 'confidence': 0.88, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [0.62, 0.32, 0.82, 0.62], 'confidence': 0.87, 'class_id': 1, 'class_name': 'car'},
    ]
    # Frame 1
    tracked_objects_1 = tracker.update(frame_1_detections)
    print("Frame 1:")
    for obj in tracked_objects_1:
        print(f"  Track ID: {obj['track_id']}, Class: {obj['class_name']}, State: {obj['state']}")
    
    # Frame 2
    tracked_objects_2 = tracker.update(frame_2_detections)
    print("\nFrame 2:")
    for obj in tracked_objects_2:
        print(f"  Track ID: {obj['track_id']}, Class: {obj['class_name']}, State: {obj['state']}")