"""Pose estimation and multi-person skeleton rendering on black backgrounds using modern MediaPipe Tasks API."""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.data.gamma import apply_gamma_correction

# Distinct color scheme for body parts (BGR format for OpenCV)
# Color distinction preserves anatomical structural features as emphasized in Garcia & SanMiguel (CVIU 2023) Fig. 2
PART_COLORS = {
    "head": (255, 0, 255),       # Magenta
    "torso": (0, 255, 0),        # Green
    "left_arm": (255, 200, 0),   # Cyan-Blue
    "right_arm": (0, 255, 255),  # Yellow
    "left_leg": (0, 0, 255),     # Red
    "right_leg": (255, 0, 100),  # Purple-Pink
}

# Anatomical connection definitions between MediaPipe 33 keypoints
CONNECTIONS = [
    # Torso
    (11, 12, PART_COLORS["torso"]),
    (11, 23, PART_COLORS["torso"]),
    (12, 24, PART_COLORS["torso"]),
    (23, 24, PART_COLORS["torso"]),
    # Head / Neck
    (0, 1, PART_COLORS["head"]),
    (1, 2, PART_COLORS["head"]),
    (2, 3, PART_COLORS["head"]),
    (0, 4, PART_COLORS["head"]),
    (4, 5, PART_COLORS["head"]),
    (5, 6, PART_COLORS["head"]),
    # Left arm
    (11, 13, PART_COLORS["left_arm"]),
    (13, 15, PART_COLORS["left_arm"]),
    # Right arm
    (12, 14, PART_COLORS["right_arm"]),
    (14, 16, PART_COLORS["right_arm"]),
    # Left leg
    (23, 25, PART_COLORS["left_leg"]),
    (25, 27, PART_COLORS["left_leg"]),
    # Right leg
    (24, 26, PART_COLORS["right_leg"]),
    (26, 28, PART_COLORS["right_leg"]),
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def ensure_pose_model(model_path: str = "models/pose_landmarker_lite.task") -> str:
    """Ensures MediaPipe Pose Landmarker task file is present on disk, downloading if necessary."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        print(f"[Pose] Downloading MediaPipe PoseLandmarker model from {MODEL_URL}...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class SkeletonRenderer:
    """Extracts multi-person poses and renders anatomical colored skeletons on a solid black canvas."""

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        num_poses: int = 4,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        line_thickness: int = 2,
        joint_radius: int = 3,
    ):
        """Initializes the multi-person skeleton renderer.

        Args:
            model_path: Path to pose_landmarker_*.task model file.
            num_poses: Maximum number of people to detect in the frame (surveillance scenes contain multiple people).
            min_pose_detection_confidence: Minimum detection confidence score.
            min_pose_presence_confidence: Minimum landmark presence score.
            line_thickness: Thickness of skeleton bone lines.
            joint_radius: Radius of joint keypoint dots.
        """
        self.num_poses = num_poses
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.landmarker = None

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            resolved_model = ensure_pose_model(model_path)
            base_options = mp.tasks.BaseOptions(model_asset_path=resolved_model)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=self.num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as e:
            print(f"[Warning] Failed to initialize MediaPipe PoseLandmarker: {e}. Falling back to blank canvas.")

    def render_on_black(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extracts all human poses and renders their colored skeletons over a black canvas.

        Args:
            frame_bgr: Single input frame in BGR format, uint8.

        Returns:
            black_canvas: Image of same shape (H, W, 3) containing only rendered skeletons.
        """
        h, w, c = frame_bgr.shape
        # Canvas strictly filled with zeros (solid black background)
        black_canvas = np.zeros((h, w, c), dtype=np.uint8)

        if self.landmarker is None:
            return black_canvas

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            results = self.landmarker.detect(mp_image)
        except Exception:
            return black_canvas

        if not results.pose_landmarks:
            return black_canvas

        # Process each detected person in the multi-person frame
        for person_landmarks in results.pose_landmarks:
            def get_pt(idx: int):
                lm = person_landmarks[idx]
                visibility = getattr(lm, "visibility", 1.0) or 1.0
                presence = getattr(lm, "presence", 1.0) or 1.0
                return (int(lm.x * w), int(lm.y * h)), min(visibility, presence)

            # Draw bones for this person
            for idx1, idx2, color in CONNECTIONS:
                pt1, score1 = get_pt(idx1)
                pt2, score2 = get_pt(idx2)
                if score1 > 0.4 and score2 > 0.4:
                    cv2.line(black_canvas, pt1, pt2, color, self.line_thickness)

            # Draw joint dots
            for idx in range(len(person_landmarks)):
                pt, score = get_pt(idx)
                if score > 0.4:
                    cv2.circle(black_canvas, pt, self.joint_radius, (255, 255, 255), -1)

        return black_canvas

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass


def process_video_skeletons(
    video_path: str,
    output_path: Optional[str] = None,
    gamma: float = 0.67,
    target_size: Optional[Tuple[int, int]] = None,
    renderer: Optional[SkeletonRenderer] = None,
) -> np.ndarray:
    """Reads a video, applies gamma correction (0.67), extracts multi-person skeletons

    on a black background, and writes the resulting skeleton video to output_path.

    Args:
        video_path: Input raw video file path.
        output_path: Path to write the output skeleton video.
        gamma: Gamma correction factor (default: 0.67).
        target_size: (H, W) tuple to resize frames.
        renderer: Optional reusable SkeletonRenderer instance.

    Returns:
        Array of rendered skeleton frames (T, H, W, 3) in RGB format, uint8.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = target_size[::-1] if target_size else (orig_w, orig_h)

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    own_renderer = False
    if renderer is None:
        renderer = SkeletonRenderer()
        own_renderer = True

    skeleton_frames: List[np.ndarray] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Resize if required
            if target_size and (frame.shape[0], frame.shape[1]) != (out_h, out_w):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            # 2. Gamma correction (contrast enhancement for surveillance scenes)
            enhanced = apply_gamma_correction(frame, gamma=gamma)

            # 3. Multi-person pose estimation on black background
            skel_bgr = renderer.render_on_black(enhanced)

            if writer:
                writer.write(skel_bgr)

            skel_rgb = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2RGB)
            skeleton_frames.append(skel_rgb)
    finally:
        cap.release()
        if writer:
            writer.release()
        if own_renderer:
            renderer.close()

    return np.array(skeleton_frames, dtype=np.uint8)
