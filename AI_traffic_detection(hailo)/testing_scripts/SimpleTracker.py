import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class SimpleTracker:
    """
    Jednoduchý object tracker založený na IOU matching.
    Funguje přímo s detekcemi z Hailo inference engine.
    """
    def __init__(self, max_lost_frames=30, iou_threshold=0.3):
        """
        Args:
            max_lost_frames: Počet snímků, než objekt "zmizí"
            iou_threshold: Threshold pro IOU matching (0-1)
        """
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        # Tracking state
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """
        Aktualizace trackeru s novými detekcemi.
        Args:
            detections: List of detections, každá detekce je dict s klíči:
                       - 'bbox': [x1, y1, x2, y2] (normalized 0-1)
                       - 'confidence': float (0-1)
                       - 'class_id': int
                       - 'class_name': str
        Returns: List of tracked objects s přidaným 'track_id'
        """
        self.frame_count += 1
        # Pokud nejsou žádné aktivní tracky, vytvoř nové
        if not self.tracks:
            return self._create_new_tracks(detections)
        # Pokud nejsou detekce, označuj tracky jako lost
        if not detections:
            self._update_lost_tracks()
            return self._get_active_tracks()
        # IOU matching mezi existujícími tracky a novými detekcemi
        matched_tracks, unmatched_detections = self._match_detections_to_tracks(detections)
        # Aktualizuj matched tracky
        for track_id, detection in matched_tracks.items():
            self.tracks[track_id].update(detection, self.frame_count)
        # Vytvoř nové tracky pro unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(track_id, detection, self.frame_count)
        # Odstraň staré ztracené tracky
        self._remove_old_tracks()
        # Vrať aktivní tracky
        return self._get_active_tracks()
    
    def _match_detections_to_tracks(self, detections):
        """
        Matchování detekcí k existujícím trackům pomocí IOU.
        Používá Hungarian algorithm pro optimální assignment.
        """
        # Připrav active tracky
        active_track_ids = [tid for tid, track in self.tracks.items() 
                           if track.is_active()]
        
        if not active_track_ids:
            return {}, detections
        
        # Vypočítej IOU matici
        iou_matrix = np.zeros((len(active_track_ids), len(detections)))
        
        for i, track_id in enumerate(active_track_ids):
            track_bbox = self.tracks[track_id].bbox
            for j, detection in enumerate(detections):
                det_bbox = detection['bbox']
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Hungarian algorithm pro matching
        # Použij -iou, protože algoritmus hledá minimum
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matched_tracks = {}
        matched_detection_indices = set()
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                track_id = active_track_ids[row]
                matched_tracks[track_id] = detections[col]
                matched_detection_indices.add(col)
        # Unmatched detections
        unmatched_detections = [det for i, det in enumerate(detections) 
                               if i not in matched_detection_indices]
        # Označit nematched tracky jako lost
        for track_id in active_track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].mark_lost(self.frame_count)
        
        return matched_tracks, unmatched_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Vypočítá Intersection over Union mezi dvěma bounding boxy.
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
    
    def _create_new_tracks(self, detections):
        """Vytvoř nové tracky pro všechny detekce."""
        tracked_objects = []
        for detection in detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(track_id, detection, self.frame_count)
            
            tracked_obj = detection.copy()
            tracked_obj['track_id'] = track_id
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def _update_lost_tracks(self):
        """Aktualizuj všechny tracky jako lost."""
        for track in self.tracks.values():
            if track.is_active():
                track.mark_lost(self.frame_count)
    
    def _remove_old_tracks(self):
        """Odstraň tracky, které jsou ztracené příliš dlouho."""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.should_remove(self.frame_count, self.max_lost_frames):
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """Vrať všechny aktivní tracky s jejich track_id."""
        tracked_objects = []
        for track in self.tracks.values():
            if track.is_active() or track.is_recently_lost(self.frame_count, frames=5):
                obj = {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'age': track.age,
                    'state': track.state
                }
                tracked_objects.append(obj)    
        return tracked_objects
    
    def reset(self):
        """Reset trackeru - vymaže všechny tracky."""
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0


class Track:
    """
    Reprezentuje jeden tracked objekt.
    """    
    STATE_NEW = 'new'
    STATE_TRACKED = 'tracked'
    STATE_LOST = 'lost'
    def __init__(self, track_id, detection, frame_count):
        """
        Args:
            track_id: Unikátní ID tracku
            detection: První detekce (dict s bbox, confidence, class_id, class_name)
            frame_count: Číslo aktuálního snímku
        """
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        
        self.state = self.STATE_NEW
        self.age = 0  # Počet snímků, kdy byl track aktivní
        self.hits = 1  # Počet successful matches
        self.time_since_update = 0  # Snímky od poslední detekce
        self.first_frame = frame_count
        self.last_seen_frame = frame_count
        
    def update(self, detection, frame_count):
        """Aktualizuj track s novou detekcí."""
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_frame = frame_count
        # Změň stav z NEW na TRACKED po 3 successful hits
        if self.state == self.STATE_NEW and self.hits >= 3:
            self.state = self.STATE_TRACKED
        elif self.state == self.STATE_LOST:
            self.state = self.STATE_TRACKED
        self.age += 1
    
    def mark_lost(self, frame_count):
        """Označ track jako ztracený."""
        self.state = self.STATE_LOST
        self.time_since_update = frame_count - self.last_seen_frame
    
    def is_active(self):
        """Je track aktivně trackovaný?"""
        return self.state in (self.STATE_NEW, self.STATE_TRACKED)
    
    def is_recently_lost(self, frame_count, frames=5):
        """Byl track ztracen nedávno (poslední N snímků)?"""
        return self.state == self.STATE_LOST and \
               (frame_count - self.last_seen_frame) <= frames
    
    def should_remove(self, frame_count, max_lost_frames):
        """Měl by být track odstraněn?"""
        # Odstraň NEW tracky, které nebyly confirmed
        if self.state == self.STATE_NEW and self.time_since_update > 3:
            return True
        
        # Odstraň LOST tracky, které jsou ztracené příliš dlouho
        if self.state == self.STATE_LOST:
            frames_lost = frame_count - self.last_seen_frame
            return frames_lost > max_lost_frames
        
        return False

if __name__ == "__main__":
    # Příklad použití s HailoInferenceEngine
    tracker = SimpleTracker(max_lost_frames=30, iou_threshold=0.3)
    # Simulace detekcí z Hailo
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