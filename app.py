"""
app.py - Flask Web Server for DLLAJ AI Crosswalk Violation Detection System

Provides REST API endpoints and WebSocket communication for real-time
video processing and violation detection display.
"""

import base64
import os
import threading

import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from detector import CrosswalkViolationDetector
from violation_tracker import ViolationTracker

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VIOLATIONS_FOLDER = os.path.join(BASE_DIR, "violations")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(VIOLATIONS_FOLDER, "data"), exist_ok=True)
os.makedirs(os.path.join(VIOLATIONS_FOLDER, "screenshots"), exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "dllaj-ai-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Initialize SocketIO with threading mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global instances
detector = CrosswalkViolationDetector()
violation_tracker = detector.violation_tracker

# Current state
current_state = {
    "video_path": None,
    "video_filename": None,
    "roi_set": False,
    "is_detecting": False,
    "detection_thread": None,
}


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==========================
# Web UI Routes
# ==========================

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


# ==========================
# REST API Endpoints
# ==========================

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload a video file for processing."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save the uploaded file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Store state
    current_state["video_path"] = filepath
    current_state["video_filename"] = filename

    # Get video info
    video_info = detector.get_video_info(filepath)

    print(f"[DLLAJ AI] Video uploaded: {filename}")
    return jsonify({
        "success": True,
        "filename": filename,
        "video_info": video_info,
    })


@app.route("/api/first-frame", methods=["POST"])
def get_first_frame():
    """Extract and return the first frame of the uploaded video."""
    if current_state["video_path"] is None:
        return jsonify({"error": "No video uploaded"}), 400

    frame = detector.get_first_frame(current_state["video_path"])
    if frame is None:
        return jsonify({"error": "Failed to extract first frame"}), 500

    # Encode frame as JPEG base64
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    video_info = detector.get_video_info(current_state["video_path"])

    return jsonify({
        "success": True,
        "image": frame_b64,
        "width": video_info.get("width", 1280),
        "height": video_info.get("height", 720),
    })


@app.route("/api/set-roi", methods=["POST"])
def set_roi():
    """Set a Region of Interest (ROI) polygon (crosswalk or centerline)."""
    data = request.get_json()
    if not data or "points" not in data:
        return jsonify({"error": "No ROI points provided"}), 400

    points = data["points"]
    zone_type = data.get("zone_type", "crosswalk")

    if len(points) < 3:
        return jsonify({"error": "ROI needs at least 3 points"}), 400

    if zone_type == "crosswalk":
        detector.set_roi(points)
        current_state["roi_set"] = True
    elif zone_type == "centerline":
        detector.set_centerline_roi(points)
        current_state["centerline_set"] = True
    else:
        return jsonify({"error": f"Invalid zone type '{zone_type}'"}), 400

    print(f"[DLLAJ AI] {zone_type.capitalize()} ROI set with {len(points)} points")
    return jsonify({
        "success": True,
        "zone_type": zone_type,
        "points_count": len(points),
    })


@app.route("/api/start-detection", methods=["POST"])
def start_detection():
    """Start the violation detection process."""
    if current_state["video_path"] is None:
        return jsonify({"error": "No video uploaded"}), 400

    if not current_state["roi_set"]:
        return jsonify({"error": "Crosswalk zone (ROI) not set"}), 400

    if current_state["is_detecting"]:
        return jsonify({"error": "Detection already in progress"}), 400

    # Get optional settings from request
    data = request.get_json() or {}
    confidence = data.get("confidence_threshold", 0.5)
    frame_skip = data.get("frame_skip", 2)

    detector.confidence_threshold = float(confidence)

    current_state["is_detecting"] = True

    return jsonify({
        "success": True,
        "message": "Detection started. Connect via WebSocket to receive frames.",
    })


@app.route("/api/stop-detection", methods=["POST"])
def stop_detection():
    """Stop the ongoing detection process."""
    detector.stop()
    current_state["is_detecting"] = False

    return jsonify({"success": True, "message": "Detection stopping..."})


@app.route("/api/violations", methods=["GET"])
def get_violations():
    """Get all recorded violations."""
    violations = violation_tracker.get_all_violations()
    return jsonify({
        "success": True,
        "violations": violations,
        "count": len(violations),
    })


@app.route("/api/violations/<violation_id>", methods=["GET"])
def get_violation(violation_id):
    """Get a specific violation by ID."""
    violation = violation_tracker.get_violation(violation_id)
    if violation is None:
        return jsonify({"error": "Violation not found"}), 404
    return jsonify({"success": True, "violation": violation})


@app.route("/api/violations/<violation_id>", methods=["DELETE"])
def delete_violation(violation_id):
    """Delete a violation record."""
    success = violation_tracker.delete_violation(violation_id)
    if not success:
        return jsonify({"error": "Violation not found"}), 404
    return jsonify({"success": True, "message": "Violation deleted"})


@app.route("/api/violations/clear", methods=["DELETE"])
def clear_violations():
    """Clear all violations."""
    violation_tracker.clear_all()
    return jsonify({"success": True, "message": "All violations cleared"})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get detection statistics."""
    return jsonify({
        "success": True,
        "detection_stats": detector.stats,
        "violation_stats": violation_tracker.get_stats(),
    })


@app.route("/violations/screenshots/<filename>")
def serve_screenshot(filename):
    """Serve violation screenshot images."""
    screenshots_dir = os.path.join(BASE_DIR, "violations", "screenshots")
    return send_from_directory(screenshots_dir, filename)


@app.route("/api/sample-videos", methods=["GET"])
def get_sample_videos():
    """List available sample videos from the sample folder."""
    sample_dir = os.path.join(BASE_DIR, "sampel pelanggaran dllaj")
    videos = []
    if os.path.isdir(sample_dir):
        for f in os.listdir(sample_dir):
            if f.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                filepath = os.path.join(sample_dir, f)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                videos.append({
                    "filename": f,
                    "path": filepath,
                    "size_mb": round(size_mb, 2),
                })
    return jsonify({"success": True, "videos": videos})


@app.route("/api/use-sample", methods=["POST"])
def use_sample_video():
    """Use a sample video from the sample folder."""
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "No filename provided"}), 400

    sample_dir = os.path.join(BASE_DIR, "sampel pelanggaran dllaj")
    filepath = os.path.join(sample_dir, data["filename"])

    if not os.path.isfile(filepath):
        return jsonify({"error": "Sample video not found"}), 404

    current_state["video_path"] = filepath
    current_state["video_filename"] = data["filename"]

    video_info = detector.get_video_info(filepath)

    print(f"[DLLAJ AI] Using sample video: {data['filename']}")
    return jsonify({
        "success": True,
        "filename": data["filename"],
        "video_info": video_info,
    })


# ==========================
# WebSocket Events
# ==========================

@socketio.on("connect")
def handle_connect():
    """Handle new WebSocket connection."""
    print(f"[DLLAJ AI] Client connected: {request.sid}")
    emit("status", {"status": "connected", "message": "Connected to DLLAJ AI server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print(f"[DLLAJ AI] Client disconnected: {request.sid}")
    # Stop detection if running
    if current_state["is_detecting"]:
        detector.stop()
        current_state["is_detecting"] = False


@socketio.on("start_detection")
def handle_start_detection(data=None):
    """Start detection via WebSocket."""
    if current_state["video_path"] is None:
        emit("error", {"message": "No video uploaded"})
        return

    if not current_state["roi_set"]:
        emit("error", {"message": "Crosswalk zone (ROI) not set"})
        return

    if current_state["is_detecting"]:
        emit("error", {"message": "Detection already in progress"})
        return

    # Parse settings
    confidence = 0.5
    frame_skip = 2
    if data:
        confidence = float(data.get("confidence_threshold", 0.5))
        frame_skip = int(data.get("frame_skip", 2))

    detector.confidence_threshold = confidence
    current_state["is_detecting"] = True
    sid = request.sid

    emit("status", {"status": "detecting", "message": "Detection started..."})

    # Run detection in a background thread
    def run_detection():
        try:
            detector.process_video(
                video_path=current_state["video_path"],
                socketio=socketio,
                sid=sid,
                frame_skip=frame_skip,
            )
        except Exception as e:
            print(f"[DLLAJ AI] Detection error: {e}")
            socketio.emit("error", {"message": f"Detection error: {str(e)}"}, room=sid)
        finally:
            current_state["is_detecting"] = False
            socketio.emit("status", {"status": "complete", "message": "Detection complete"}, room=sid)

    thread = threading.Thread(target=run_detection, daemon=True)
    thread.start()
    current_state["detection_thread"] = thread


@socketio.on("stop_detection")
def handle_stop_detection():
    """Stop detection via WebSocket."""
    detector.stop()
    current_state["is_detecting"] = False
    emit("status", {"status": "stopped", "message": "Detection stopped"})


# ==========================
# Main Entry Point
# ==========================

if __name__ == "__main__":
    print("=" * 60)
    print("  DLLAJ AI - Crosswalk Violation Detection System")
    print("  Based on UU Nomor 22 Tahun 2009")
    print("=" * 60)
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print(f"  Violations folder: {VIOLATIONS_FOLDER}")
    print()

    # Pre-load the YOLO model
    print("[DLLAJ AI] Loading YOLOv8 model...")
    try:
        detector.load_model()
        print("[DLLAJ AI] Model loaded successfully!")
    except Exception as e:
        print(f"[DLLAJ AI] Warning: Could not pre-load model: {e}")
        print("[DLLAJ AI] Model will be loaded on first detection.")

    print()
    print("[DLLAJ AI] Starting web server on http://localhost:5001")
    print()

    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
