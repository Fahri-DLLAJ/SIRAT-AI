"""
violation_tracker.py - Violation Data Management for DLLAJ AI

Handles saving, loading, and managing crosswalk violation records.
Violations are stored as JSON and screenshots are saved as PNG files.
"""

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

# Indonesia timezone (WIB, UTC+7)
WIB = timezone(timedelta(hours=7))

# Violation type definitions based on UU No. 22 Tahun 2009
VIOLATION_TYPES = {
    "vehicle_on_crosswalk": {
        "name": "Kendaraan Berhenti di Zebra Cross",
        "description_template": "Kendaraan bermotor ({vehicle_type}) berhenti atau melintas di area zebra cross saat pejalan kaki sedang menyeberang",
        "legal_reference": "Pasal 106 ayat (2) jo. Pasal 131 ayat (2) UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 2 bulan atau denda paling banyak Rp500.000",
    },
    "not_yielding_to_pedestrian": {
        "name": "Tidak Mengutamakan Pejalan Kaki",
        "description_template": "Kendaraan bermotor ({vehicle_type}) tidak mengutamakan keselamatan pejalan kaki di area penyeberangan",
        "legal_reference": "Pasal 106 ayat (2) jo. Pasal 284 UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 2 bulan atau denda paling banyak Rp500.000",
    },
    "running_red_light": {
        "name": "Melanggar Lampu Merah di Zebra Cross",
        "description_template": "Kendaraan bermotor ({vehicle_type}) melanggar isyarat lampu merah di area penyeberangan",
        "legal_reference": "Pasal 106 ayat (4) huruf c jo. Pasal 287 ayat (2) UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 2 bulan atau denda paling banyak Rp500.000",
    },
    "not_slowing_down": {
        "name": "Tidak Memperlambat Kendaraan",
        "description_template": "Kendaraan bermotor ({vehicle_type}) tidak memperlambat kendaraan saat mendekati area penyeberangan dengan pejalan kaki",
        "legal_reference": "Pasal 116 ayat (2) huruf f UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 1 bulan atau denda paling banyak Rp250.000",
    },
    "parked_on_crosswalk": {
        "name": "Parkir/Berhenti di Zebra Cross",
        "description_template": "Kendaraan bermotor ({vehicle_type}) parkir atau berhenti di area penyeberangan pejalan kaki",
        "legal_reference": "Pasal 118 huruf b UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 1 bulan atau denda paling banyak Rp250.000",
    },
    "crossing_center_line": {
        "name": "Melanggar Marka Jalan (Melintasi Garis Tengah)",
        "description_template": "{vehicle_type} melanggar marka jalan dengan melintasi atau berada di garis tengah jalan",
        "legal_reference": "Pasal 106 ayat (4) huruf a jo. Pasal 287 ayat (1) UU No. 22 Tahun 2009",
        "penalty": "Pidana kurungan paling lama 2 bulan atau denda paling banyak Rp500.000",
    },
    "priority_vehicle": {
        "name": "Kendaraan Prioritas Melintas",
        "description_template": "Kendaraan prioritas ({vehicle_type}) terdeteksi melintas",
        "legal_reference": "Pasal 134 UU No. 22 Tahun 2009 (Hak Utama Pengguna Jalan)",
        "penalty": "Wajib didahului dan diberi kelonggaran jalan",
    },
}

# YOLO class names to Indonesian vehicle types
VEHICLE_TYPE_NAMES = {
    "car": "Mobil",
    "truck": "Truk",
    "bus": "Bus",
    "motorcycle": "Sepeda Motor",
    "bicycle": "Sepeda",
    "pedestrian": "Pejalan Kaki",
    "pejalan kaki": "Pejalan Kaki",
    "priority": "Ambulans/Pemadam Kebakaran",
}


class ViolationTracker:
    """Manages violation records and screenshots for the DLLAJ AI system."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "violations" / "data"
        self.screenshots_dir = self.base_dir / "violations" / "screenshots"
        self.violations_file = self.data_dir / "violations.json"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Load existing violations
        self.violations = self._load_violations()

    def _load_violations(self) -> list:
        """Load violations from the JSON file."""
        if self.violations_file.exists():
            try:
                with open(self.violations_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_violations(self):
        """Save all violations to the JSON file."""
        with open(self.violations_file, "w", encoding="utf-8") as f:
            json.dump(self.violations, f, indent=2, ensure_ascii=False)

    def _generate_id(self) -> str:
        """Generate a unique violation ID."""
        now = datetime.now(WIB)
        date_str = now.strftime("%Y%m%d-%H%M%S")
        short_uuid = uuid.uuid4().hex[:4].upper()
        return f"VIO-{date_str}-{short_uuid}"

    def _format_video_time(self, frame_number: int, fps: float) -> str:
        """Convert frame number to video time string MM:SS.mmm."""
        if fps <= 0:
            return "00:00.000"
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    def save_screenshot(self, frame: np.ndarray, violation_id: str) -> str:
        """
        Save a violation screenshot.

        Args:
            frame: The OpenCV frame (BGR numpy array) to save.
            violation_id: The violation ID for the filename.

        Returns:
            Relative path to the saved screenshot.
        """
        filename = f"{violation_id}.png"
        filepath = self.screenshots_dir / filename
        cv2.imwrite(str(filepath), frame)
        return f"violations/screenshots/{filename}"

    def record_violation(
        self,
        frame: np.ndarray,
        violation_type: str,
        vehicle_type: str,
        vehicle_bbox: list,
        confidence: float,
        frame_number: int,
        fps: float,
        video_filename: str,
        pedestrian_detected: bool = True,
        annotated_frame: np.ndarray = None,
    ) -> dict:
        """
        Record a new violation with screenshot and metadata.

        Args:
            frame: Original video frame (or annotated frame for screenshot).
            violation_type: Key from VIOLATION_TYPES dict.
            vehicle_type: Vehicle class name (car, truck, bus, motorcycle, etc.).
            vehicle_bbox: [x1, y1, x2, y2] bounding box coordinates.
            confidence: Detection confidence score (0-1).
            frame_number: Frame number in the video.
            fps: Video frames per second.
            video_filename: Name of the source video file.
            pedestrian_detected: Whether a pedestrian was detected nearby.
            annotated_frame: Frame with bounding boxes drawn (used for screenshot).

        Returns:
            The violation record dict.
        """
        violation_id = self._generate_id()

        # Use annotated frame for screenshot if available, otherwise original frame
        screenshot_frame = annotated_frame if annotated_frame is not None else frame
        screenshot_path = self.save_screenshot(screenshot_frame, violation_id)

        # Get violation type info
        type_info = VIOLATION_TYPES.get(violation_type, VIOLATION_TYPES["vehicle_on_crosswalk"])
        vehicle_type_id = vehicle_type.lower() if vehicle_type else "car"
        vehicle_type_name = VEHICLE_TYPE_NAMES.get(vehicle_type_id, vehicle_type_id.capitalize())

        violation = {
            "id": violation_id,
            "timestamp": datetime.now(WIB).isoformat(),
            "video_file": video_filename,
            "frame_number": frame_number,
            "time_in_video": self._format_video_time(frame_number, fps),
            "violation_type": violation_type,
            "violation_name": type_info["name"],
            "description": type_info["description_template"].format(vehicle_type=vehicle_type_name),
            "legal_reference": type_info["legal_reference"],
            "penalty": type_info["penalty"],
            "vehicle_type": vehicle_type_name,
            "vehicle_bbox": [int(x) for x in vehicle_bbox],
            "pedestrian_detected": pedestrian_detected,
            "confidence": round(confidence, 4),
            "screenshot_path": screenshot_path,
        }

        self.violations.append(violation)
        self._save_violations()

        return violation

    def get_all_violations(self) -> list:
        """Get all violation records."""
        return self.violations

    def get_violation(self, violation_id: str) -> dict | None:
        """Get a specific violation by ID."""
        for v in self.violations:
            if v["id"] == violation_id:
                return v
        return None

    def delete_violation(self, violation_id: str) -> bool:
        """Delete a violation record and its screenshot."""
        for i, v in enumerate(self.violations):
            if v["id"] == violation_id:
                # Delete screenshot file
                screenshot_path = self.base_dir / v.get("screenshot_path", "")
                if screenshot_path.exists():
                    screenshot_path.unlink()

                # Remove from list
                self.violations.pop(i)
                self._save_violations()
                return True
        return False

    def clear_all(self):
        """Clear all violations and screenshots."""
        # Delete all screenshot files
        for screenshot_file in self.screenshots_dir.glob("*.png"):
            screenshot_file.unlink()

        self.violations = []
        self._save_violations()

    def get_stats(self) -> dict:
        """Get summary statistics of detected violations."""
        stats = {
            "total_violations": len(self.violations),
            "by_type": {},
            "by_vehicle": {},
        }

        for v in self.violations:
            vtype = v.get("violation_type", "unknown")
            veh = v.get("vehicle_type", "Unknown")

            stats["by_type"][vtype] = stats["by_type"].get(vtype, 0) + 1
            stats["by_vehicle"][veh] = stats["by_vehicle"].get(veh, 0) + 1

        return stats
