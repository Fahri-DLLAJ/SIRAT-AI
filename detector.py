"""
detector.py - Crosswalk Violation Detection Engine for DLLAJ AI

Uses YOLOv8 for real-time object detection (vehicles, pedestrians)
and checks for violations in user-defined crosswalk zones (ROI).
"""

import time
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

from violation_tracker import ViolationTracker


# COCO class IDs for relevant objects
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

# All relevant class IDs
RELEVANT_CLASS_IDS = {PERSON_CLASS_ID} | set(VEHICLE_CLASS_IDS.keys())

# Minimum overlap ratio between vehicle bbox and crosswalk zone to trigger violation
MIN_OVERLAP_RATIO = 0.01

# Minimum frames a vehicle must be in the crosswalk zone to count as a violation
MIN_VIOLATION_FRAMES = 1

# Cooldown frames between recording violations for the same tracked vehicle
VIOLATION_COOLDOWN_FRAMES = 90  # ~3 seconds at 30fps

# Colors for drawing (BGR format for OpenCV)
COLORS = {
    "vehicle_normal": (0, 255, 0),       # Green - vehicle not in violation
    "vehicle_violation": (0, 0, 255),     # Red - vehicle in violation
    "pedestrian": (255, 191, 0),          # Cyan/light blue - pedestrian
    "crosswalk_zone": (0, 255, 255),      # Yellow - crosswalk ROI
    "crosswalk_fill": (0, 200, 200),      # Slightly darker yellow for fill
    "centerline_zone": (255, 0, 127),     # Pink/Purple - centerline ROI
    "centerline_fill": (200, 0, 100),     # Slightly darker pink/purple for fill
    "text_bg": (0, 0, 0),                 # Black - text background
    "text_fg": (255, 255, 255),           # White - text foreground
    "violation_alert": (0, 0, 255),       # Red - violation alert
}


class SimpleTracker:
    """
    Simple centroid-based object tracker.
    Tracks objects across frames using centroid distance matching.
    """

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}          # track_id -> centroid (cx, cy)
        self.bboxes = {}           # track_id -> (x1, y1, x2, y2)
        self.disappeared = {}      # track_id -> frames since last seen
        self.history = defaultdict(list)  # track_id -> list of centroid coordinates
        self.max_disappeared = max_disappeared

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of (x1, y1, x2, y2, class_id, confidence)

        Returns:
            dict: track_id -> (x1, y1, x2, y2, class_id, confidence)
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return {}

        # Get centroids of new detections
        new_centroids = []
        for det in detections:
            new_centroids.append(self._get_centroid(det[:4]))

        # If no existing tracks, register all
        if len(self.objects) == 0:
            result = {}
            for i, det in enumerate(detections):
                track_id = self._register(new_centroids[i], det[:4])
                result[track_id] = det
            return result

        # Match existing tracks to new detections using distance
        existing_ids = list(self.objects.keys())
        existing_centroids = [self.objects[tid] for tid in existing_ids]

        # Calculate distance matrix
        dist_matrix = np.zeros((len(existing_centroids), len(new_centroids)))
        for i, ec in enumerate(existing_centroids):
            for j, nc in enumerate(new_centroids):
                dist_matrix[i, j] = np.sqrt((ec[0] - nc[0]) ** 2 + (ec[1] - nc[1]) ** 2)

        # Greedy matching: assign each existing track to closest detection
        used_det = set()
        used_track = set()
        result = {}

        # Sort by distance and match greedily
        matches = []
        for i in range(len(existing_centroids)):
            for j in range(len(new_centroids)):
                matches.append((dist_matrix[i, j], i, j))
        matches.sort()

        for dist, track_idx, det_idx in matches:
            if track_idx in used_track or det_idx in used_det:
                continue
            if dist > 150:  # Max matching distance in pixels
                continue
            used_track.add(track_idx)
            used_det.add(det_idx)
            track_id = existing_ids[track_idx]
            self.objects[track_id] = new_centroids[det_idx]
            self.bboxes[track_id] = detections[det_idx][:4]
            self.disappeared[track_id] = 0
            
            # Record history
            self.history[track_id].append(new_centroids[det_idx])
            if len(self.history[track_id]) > 30:
                self.history[track_id].pop(0)
                
            result[track_id] = detections[det_idx]

        # Handle unmatched existing tracks
        for i, tid in enumerate(existing_ids):
            if i not in used_track:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        # Register new detections that weren't matched
        for j, det in enumerate(detections):
            if j not in used_det:
                track_id = self._register(new_centroids[j], det[:4])
                result[track_id] = det

        return result

    def _register(self, centroid, bbox):
        track_id = self.next_id
        self.objects[track_id] = centroid
        self.bboxes[track_id] = bbox
        self.disappeared[track_id] = 0
        self.history[track_id] = [centroid]
        self.next_id += 1
        return track_id

    def _deregister(self, track_id):
        del self.objects[track_id]
        del self.bboxes[track_id]
        del self.disappeared[track_id]
        if track_id in self.history:
            del self.history[track_id]


class CrosswalkViolationDetector:
    """
    Main detection engine for crosswalk violations.
    Uses YOLOv8 for object detection and polygon-based ROI for crosswalk zone.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.model = None
        self.roi_polygon = None         # Shapely Polygon for the crosswalk zone
        self.roi_points = []            # List of (x, y) points for drawing
        self.centerline_polygon = None  # Shapely Polygon for the center line zone
        self.centerline_points = []     # List of (x, y) points for drawing
        self.confidence_threshold = confidence_threshold
        self.tracker = SimpleTracker(max_disappeared=30)
        self.pedestrian_tracker = SimpleTracker(max_disappeared=30)
        self.violation_tracker = ViolationTracker()

        # Track vehicle presence in zones
        self.vehicle_in_zone_frames = defaultdict(int)  # track_id -> consecutive frames in zone
        self.vehicle_in_centerline_frames = defaultdict(int) # track_id -> consecutive frames in centerline
        self.vehicle_violation_cooldown = defaultdict(int)  # track_id -> cooldown counter

        # Detection stats
        self.stats = {
            "total_vehicles": 0,
            "total_pedestrians": 0,
            "total_violations": 0,
            "frames_processed": 0,
        }

        # State flags
        self.is_running = False
        self._should_stop = False

    def load_model(self):
        """Load the YOLOv8 model."""
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")  # Nano model for speed
        print("[DLLAJ AI] YOLOv8n model loaded successfully")

    def set_roi(self, points: list):
        """
        Set the crosswalk Region of Interest (ROI) polygon.

        Args:
            points: List of [x, y] coordinate pairs defining the polygon vertices.
        """
        self.roi_points = [(int(p[0]), int(p[1])) for p in points]
        if len(self.roi_points) >= 3:
            self.roi_polygon = Polygon(self.roi_points)
            print(f"[DLLAJ AI] ROI set with {len(self.roi_points)} points, area: {self.roi_polygon.area:.0f}px²")
        else:
            self.roi_polygon = None
            print("[DLLAJ AI] ROI needs at least 3 points")

    def _check_overlap(self, bbox) -> float:
        """
        Check how much a bounding box overlaps with the crosswalk ROI.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Overlap ratio (0.0 - 1.0) of bbox area intersecting with ROI.
        """
        if self.roi_polygon is None:
            return 0.0

        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        det_box = shapely_box(x1, y1, x2, y2)

        if not self.roi_polygon.intersects(det_box):
            return 0.0

        intersection = self.roi_polygon.intersection(det_box)
        bbox_area = det_box.area
        if bbox_area == 0:
            return 0.0

        return intersection.area / bbox_area

    def set_centerline_roi(self, points: list):
        """
        Set the centerline Region of Interest (ROI) polygon.

        Args:
            points: List of [x, y] coordinate pairs defining the polygon vertices.
        """
        self.centerline_points = [(int(p[0]), int(p[1])) for p in points]
        if len(self.centerline_points) >= 3:
            self.centerline_polygon = Polygon(self.centerline_points)
            print(f"[DLLAJ AI] Centerline ROI set with {len(self.centerline_points)} points, area: {self.centerline_polygon.area:.0f}px²")
        else:
            self.centerline_polygon = None
            print("[DLLAJ AI] Centerline ROI needs at least 3 points")

    def _check_centerline_overlap(self, bbox) -> float:
        """
        Check how much a bounding box overlaps with the centerline ROI.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Overlap ratio (0.0 - 1.0) of bbox area intersecting with centerline ROI.
        """
        if self.centerline_polygon is None:
            return 0.0

        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        det_box = shapely_box(x1, y1, x2, y2)

        if not self.centerline_polygon.intersects(det_box):
            return 0.0

        intersection = self.centerline_polygon.intersection(det_box)
        bbox_area = det_box.area
        if bbox_area == 0:
            return 0.0

        return intersection.area / bbox_area

    def _determine_violation_type(self, track_id: int, overlap: float, pedestrian_nearby: bool, frames_in_zone: int) -> str | None:
        """
        Determine the type of violation based on context.

        Args:
            track_id: Vehicle track ID.
            overlap: Overlap ratio of vehicle with crosswalk zone.
            pedestrian_nearby: Whether a pedestrian is detected near the crosswalk.
            frames_in_zone: Consecutive frames the vehicle has been in the zone.

        Returns:
            Violation type string or None if no violation.
        """
        if overlap < MIN_OVERLAP_RATIO:
            return None

        # Vehicle overlapping with crosswalk zone
        if pedestrian_nearby:
            if frames_in_zone > 15:
                # Vehicle stopped/parked on crosswalk with pedestrians present
                return "vehicle_on_crosswalk"
            elif frames_in_zone >= 1:
                # Vehicle immediately not yielding to pedestrians
                return "not_yielding_to_pedestrian"
        else:
            if frames_in_zone > 5:
                # Vehicle parked/stopped on crosswalk without pedestrians
                return "parked_on_crosswalk"

        return None

    def _is_pedestrian_near_crosswalk(self, pedestrian_bboxes: list) -> bool:
        """Check if any pedestrian is near the crosswalk zone."""
        if self.roi_polygon is None:
            return False

        for bbox in pedestrian_bboxes:
            overlap = self._check_overlap(bbox)
            if overlap > 0.05:  # Pedestrian even slightly in/near crosswalk
                return True

            # Also check if pedestrian centroid is within expanded ROI
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            expanded_roi = self.roi_polygon.buffer(50)  # 50px buffer
            from shapely.geometry import Point
            if expanded_roi.contains(Point(cx, cy)):
                return True

        return False

    def _draw_annotations(self, frame: np.ndarray, tracked_objects: dict,
                          tracked_pedestrians: dict, violations_this_frame: list) -> np.ndarray:
        """
        Draw bounding boxes, labels, and detection zones on the frame.

        Args:
            frame: Original frame to annotate.
            tracked_objects: dict of track_id -> detection info.
            tracked_pedestrians: dict of track_id -> pedestrian detection info.
            violations_this_frame: list of violation info dicts for this frame.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw crosswalk zone (ROI polygon)
        if self.roi_points and len(self.roi_points) >= 3:
            pts = np.array(self.roi_points, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], COLORS["crosswalk_fill"])
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            cv2.polylines(annotated, [pts], True, COLORS["crosswalk_zone"], 2, cv2.LINE_AA)

            # Label the zone
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(annotated, "ZEBRA CROSS ZONE", (centroid[0] - 80, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["crosswalk_zone"], 2, cv2.LINE_AA)

        # Draw center line zone (ROI polygon)
        if self.centerline_points and len(self.centerline_points) >= 3:
            pts = np.array(self.centerline_points, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], COLORS["centerline_fill"])
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            cv2.polylines(annotated, [pts], True, COLORS["centerline_zone"], 2, cv2.LINE_AA)

            # Label the zone
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(annotated, "CENTER LINE ZONE", (centroid[0] - 80, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["centerline_zone"], 2, cv2.LINE_AA)

        # Draw vehicle bounding boxes
        violation_track_ids = {v["track_id"] for v in violations_this_frame}
        for track_id, det in tracked_objects.items():
            x1, y1, x2, y2 = [int(c) for c in det[:4]]
            class_id = int(det[4])
            conf = float(det[5])

            if class_id in VEHICLE_CLASS_IDS:
                vehicle_type = VEHICLE_CLASS_IDS[class_id]
                is_violating = track_id in violation_track_ids
                color = COLORS["vehicle_violation"] if is_violating else COLORS["vehicle_normal"]

                # Draw bbox
                thickness = 3 if is_violating else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Draw label
                if is_violating:
                    label = f"VIOLATION! {vehicle_type} #{track_id} ({conf:.2f})"
                else:
                    label = f"vehicle #{track_id} ({conf:.2f})"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1, cv2.LINE_AA)

                # Draw violation flash effect
                if is_violating:
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                                  COLORS["violation_alert"], 4)
                    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Draw pedestrian bounding boxes
        for track_id, det in tracked_pedestrians.items():
            x1, y1, x2, y2 = [int(c) for c in det[:4]]
            conf = float(det[5]) if len(det) > 5 else 0.0
            
            is_violating = f"p_{track_id}" in violation_track_ids
            color = COLORS["vehicle_violation"] if is_violating else COLORS["pedestrian"]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if is_violating else 2)
            if is_violating:
                label = f"VIOLATION! Pedestrian #{track_id} ({conf:.2f})"
            else:
                label = f"vehicle #{track_id} ({conf:.2f})"
                
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"] if is_violating else (0, 0, 0), 1, cv2.LINE_AA)
            
            if is_violating:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), COLORS["violation_alert"], 4)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Draw violation alert banner at top if any violations this frame
        if violations_this_frame:
            banner_h = 40
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            text = f"⚠ VIOLATION DETECTED: {len(violations_this_frame)} object(s) in zones"
            cv2.putText(annotated, text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw info bar at bottom
        info_h = 30
        cv2.rectangle(annotated, (0, h - info_h), (w, h), (0, 0, 0), -1)
        info_text = (f"Frame: {self.stats['frames_processed']} | "
                     f"Vehicles: {self.stats['total_vehicles']} | "
                     f"Pedestrians: {self.stats['total_pedestrians']} | "
                     f"Violations: {self.stats['total_violations']}")
        cv2.putText(annotated, info_text, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated

    def process_video(self, video_path: str, socketio=None, sid=None,
                      frame_skip: int = 2):
        """
        Process a video file and detect crosswalk violations.

        Args:
            video_path: Path to the video file.
            socketio: Flask-SocketIO instance for emitting events.
            sid: Socket session ID.
            frame_skip: Process every Nth frame for speed (1 = every frame).

        Yields:
            Tuples of (annotated_frame_bytes, violation_data_or_none)
        """
        if self.model is None:
            self.load_model()

        if self.roi_polygon is None:
            print("[DLLAJ AI] Warning: No ROI set. Violations will not be detected.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[DLLAJ AI] Error: Cannot open video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_filename = video_path.split("/")[-1] if "/" in video_path else video_path.split("\\")[-1]

        print(f"[DLLAJ AI] Processing video: {video_filename}")
        print(f"[DLLAJ AI] Resolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")

        self.is_running = True
        self._should_stop = False
        self.stats = {
            "total_vehicles": 0,
            "total_pedestrians": 0,
            "total_violations": 0,
            "frames_processed": 0,
        }

        # Reset trackers
        self.tracker = SimpleTracker(max_disappeared=30)
        self.pedestrian_tracker = SimpleTracker(max_disappeared=30)
        self.vehicle_in_zone_frames.clear()
        self.vehicle_in_centerline_frames.clear()
        self.vehicle_violation_cooldown.clear()

        # Sets to store unique track IDs seen so far in this video run
        self.unique_vehicle_ids = set()
        self.unique_pedestrian_ids = set()

        frame_count = 0
        last_emit_time = 0

        try:
            while cap.isOpened() and not self._should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    continue

                self.stats["frames_processed"] = frame_count

                # Run YOLO detection
                results = self.model(frame, conf=self.confidence_threshold,
                                     classes=list(RELEVANT_CLASS_IDS),
                                     verbose=False)

                # Parse detections
                vehicle_detections = []
                pedestrian_bboxes = []

                if results and len(results) > 0:
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box_data in result.boxes:
                            x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
                            conf = float(box_data.conf[0])
                            class_id = int(box_data.cls[0])

                            if class_id == PERSON_CLASS_ID:
                                pedestrian_bboxes.append([x1, y1, x2, y2, class_id, conf])
                            elif class_id in VEHICLE_CLASS_IDS:
                                vehicle_detections.append([x1, y1, x2, y2, class_id, conf])

                # Filter out pedestrians that overlap with vehicles (to avoid double counting riders)
                filtered_pedestrians = []
                for p_box in pedestrian_bboxes:
                    px1, py1, px2, py2 = p_box[:4]
                    is_overlapping = False
                    for v_box in vehicle_detections:
                        vx1, vy1, vx2, vy2 = v_box[:4]
                        # Check intersection over pedestrian bounding box area
                        xi1 = max(px1, vx1)
                        yi1 = max(py1, vy1)
                        xi2 = min(px2, vx2)
                        yi2 = min(py2, vy2)
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        p_area = (px2 - px1) * (py2 - py1)
                        if p_area > 0 and (inter / p_area) > 0.5:
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        filtered_pedestrians.append(p_box)

                # Track vehicles
                tracked = self.tracker.update(vehicle_detections)
                for track_id in tracked.keys():
                    self.unique_vehicle_ids.add(track_id)

                # Track pedestrians
                tracked_pedestrians = self.pedestrian_tracker.update(filtered_pedestrians)

                # Determine actual pedestrians vs. upgraded motorcycles
                actual_pedestrians = {}
                upgraded_motorcycles = {}

                for track_id, det in tracked_pedestrians.items():
                    px1, py1, px2, py2 = det[:4]
                    p_class_id = int(det[4])
                    pconf = float(det[5])

                    # Calculate speed using tracker history
                    history = self.pedestrian_tracker.history.get(track_id, [])
                    is_rider = False
                    if len(history) >= 3:
                        dx = history[-1][0] - history[0][0]
                        dy = history[-1][1] - history[0][1]
                        speed = np.sqrt(dx**2 + dy**2) / len(history)
                        if speed > 1.5:  # Moving at vehicle speeds
                            is_rider = True

                    if is_rider:
                        # Upgrade to a motorcycle
                        upgraded_motorcycles[f"up_{track_id}"] = [px1, py1, px2, py2, 3, pconf]
                        if track_id in self.unique_pedestrian_ids:
                            self.unique_pedestrian_ids.remove(track_id)
                    else:
                        actual_pedestrians[track_id] = det
                        self.unique_pedestrian_ids.add(track_id)

                self.stats["total_pedestrians"] = len(self.unique_pedestrian_ids)

                # Merge upgraded motorcycles into tracked vehicles
                for up_id, det in upgraded_motorcycles.items():
                    tracked[up_id] = det
                    self.unique_vehicle_ids.add(up_id)
                self.stats["total_vehicles"] = len(self.unique_vehicle_ids)

                # Check for violations
                violations_this_frame = []
                pedestrian_nearby = self._is_pedestrian_near_crosswalk(pedestrian_bboxes)

                for track_id, det in tracked.items():
                    x1, y1, x2, y2 = det[:4]
                    class_id = int(det[4])
                    conf = float(det[5])
                    vehicle_type = VEHICLE_CLASS_IDS.get(class_id, "car")

                    # Check overlap with crosswalk zone
                    overlap = self._check_overlap([x1, y1, x2, y2])
                    if overlap >= MIN_OVERLAP_RATIO:
                        self.vehicle_in_zone_frames[track_id] += 1
                    else:
                        self.vehicle_in_zone_frames[track_id] = max(0, self.vehicle_in_zone_frames.get(track_id, 0) - 1)

                    # Check overlap with centerline zone
                    centerline_overlap = self._check_centerline_overlap([x1, y1, x2, y2])
                    if centerline_overlap >= MIN_OVERLAP_RATIO:
                        self.vehicle_in_centerline_frames[track_id] += 1
                    else:
                        self.vehicle_in_centerline_frames[track_id] = max(0, self.vehicle_in_centerline_frames.get(track_id, 0) - 1)

                    # Decrease cooldown
                    if track_id in self.vehicle_violation_cooldown:
                        self.vehicle_violation_cooldown[track_id] -= 1
                        if self.vehicle_violation_cooldown[track_id] <= 0:
                            del self.vehicle_violation_cooldown[track_id]

                    # Determine violation
                    frames_in_zone = self.vehicle_in_zone_frames.get(track_id, 0)
                    frames_in_centerline = self.vehicle_in_centerline_frames.get(track_id, 0)
                    
                    violation_type = self._determine_violation_type(
                        track_id, overlap, pedestrian_nearby, frames_in_zone
                    )

                    # Check centerline violation if no crosswalk violation
                    if not violation_type and centerline_overlap >= MIN_OVERLAP_RATIO:
                        # Determine direction of travel using centroid history
                        is_moving_towards_camera = True
                        history = self.tracker.history.get(track_id, [])
                        if len(history) >= 3:
                            dy = history[-1][1] - history[0][1]
                            if dy < -2:  # Moving away from camera is correct direction for left lane
                                is_moving_towards_camera = False
                                
                        if is_moving_towards_camera and frames_in_centerline >= 1:  # Overlap centerline for >= 1 frame
                            violation_type = "crossing_center_line"

                    if violation_type and track_id not in self.vehicle_violation_cooldown:
                        violations_this_frame.append({
                            "track_id": track_id,
                            "violation_type": violation_type,
                            "vehicle_type": vehicle_type,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "overlap": max(overlap, centerline_overlap),
                        })

                # Check for pedestrian violations
                for track_id, det in actual_pedestrians.items():
                    px1, py1, px2, py2 = det[:4]
                    pconf = float(det[5]) if len(det) > 5 else 0.0
                    
                    # Check overlap with centerline zone
                    p_centerline_overlap = self._check_centerline_overlap([px1, py1, px2, py2])
                    cooldown_id = f"p_{track_id}"
                    
                    if p_centerline_overlap >= MIN_OVERLAP_RATIO:
                        self.vehicle_in_centerline_frames[cooldown_id] += 1
                    else:
                        self.vehicle_in_centerline_frames[cooldown_id] = max(0, self.vehicle_in_centerline_frames.get(cooldown_id, 0) - 1)
                        
                    # Decrease cooldown
                    if cooldown_id in self.vehicle_violation_cooldown:
                        self.vehicle_violation_cooldown[cooldown_id] -= 1
                        if self.vehicle_violation_cooldown[cooldown_id] <= 0:
                            del self.vehicle_violation_cooldown[cooldown_id]
                            
                    p_frames_in_centerline = self.vehicle_in_centerline_frames.get(cooldown_id, 0)
                    if p_centerline_overlap >= MIN_OVERLAP_RATIO and p_frames_in_centerline >= 1:
                        if cooldown_id not in self.vehicle_violation_cooldown:
                            violations_this_frame.append({
                                "track_id": cooldown_id,
                                "violation_type": "crossing_center_line",
                                "vehicle_type": "pedestrian",
                                "bbox": [px1, py1, px2, py2],
                                "confidence": pconf,
                                "overlap": p_centerline_overlap,
                            })

                # Draw annotations
                annotated_frame = self._draw_annotations(
                    frame, tracked, tracked_pedestrians, violations_this_frame
                )

                # Record violations
                for v_info in violations_this_frame:
                    violation_record = self.violation_tracker.record_violation(
                        frame=frame,
                        violation_type=v_info["violation_type"],
                        vehicle_type=v_info["vehicle_type"],
                        vehicle_bbox=v_info["bbox"],
                        confidence=v_info["confidence"],
                        frame_number=frame_count,
                        fps=fps,
                        video_filename=video_filename,
                        pedestrian_detected=pedestrian_nearby,
                        annotated_frame=annotated_frame,
                    )
                    self.stats["total_violations"] += 1

                    # Set cooldown for this tracked vehicle
                    self.vehicle_violation_cooldown[v_info["track_id"]] = VIOLATION_COOLDOWN_FRAMES

                    # Emit violation event via WebSocket
                    if socketio and sid:
                        socketio.emit("violation", violation_record, room=sid)

                # Encode annotated frame as JPEG for streaming
                _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()

                # Emit frame via WebSocket (rate limited to ~15fps for streaming)
                current_time = time.time()
                if socketio and sid and (current_time - last_emit_time) >= 0.066:
                    import base64
                    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                    socketio.emit("frame", {
                        "image": frame_b64,
                        "frame_number": frame_count,
                        "total_frames": total_frames,
                        "progress": round(frame_count / max(total_frames, 1) * 100, 1),
                    }, room=sid)

                    socketio.emit("progress", {
                        "current": frame_count,
                        "total": total_frames,
                        "percent": round(frame_count / max(total_frames, 1) * 100, 1),
                        "stats": self.stats,
                    }, room=sid)

                    last_emit_time = current_time
                    socketio.sleep(0)  # Yield to allow event loop processing

        finally:
            cap.release()
            self.is_running = False

            # Emit completion
            if socketio and sid:
                socketio.emit("detection_complete", {
                    "stats": self.stats,
                    "violations": self.violation_tracker.get_all_violations(),
                }, room=sid)

            print(f"[DLLAJ AI] Detection complete. Processed {frame_count} frames.")
            print(f"[DLLAJ AI] Found {self.stats['total_violations']} violations.")

    def stop(self):
        """Request detection to stop."""
        self._should_stop = True

    def get_first_frame(self, video_path: str) -> np.ndarray | None:
        """Extract the first frame from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()
        return info
