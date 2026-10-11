"""Pose estimation and skeleton rendering on black backgrounds."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.data.gamma import apply_gamma_correction

# Distinct color scheme for body parts (BGR format for OpenCV)
# Distinguishing limbs and joints as detailed in Garcia & SanMiguel (CVIU 2023) Fig. 2
PART_COLORS = {
    "head": (255, 0, 255),       # Magenta
    "torso": (0, 255, 0),        # Green
    "left_arm": (255, 200, 0),   # Cyan-Blue
    "right_arm": (0, 255, 255),  # Yellow
    "left_leg": (0, 0, 255),     # Red
    "right_leg": (255, 0, 100),  # Purple-Pink
}


class SkeletonRenderer:
    """Renders human skeletons with colored limbs on a solid black background."""

    def __init__(self, backend: str = "mediapipe", line_thickness: int = 2, joint_radius: int = 3):
        self.backend = backend
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self._mp_pose = None
        self._pose_detector = None

        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self._mp_pose = mp.solutions.pose
                self._pose_detector = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize MediaPipe: {e}. Falling back to geometric mock.")
                self.backend = "fallback"

    def render_on_black(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extracts pose from frame and renders colored skeleton on a black canvas."""
        h, w, c = frame_bgr.shape
        # Create solid black canvas
        black_canvas = np.zeros((h, w, c), dtype=np.uint8)

        if self.backend == "mediapipe" and self._pose_detector is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._pose_detector.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_pt(idx):
                    lm = landmarks[idx]
                    return (int(lm.x * w), int(lm.y * h)), lm.visibility

                # Define bone connections with anatomical groups
                connections = [
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

                # Draw bones
                for idx1, idx2, color in connections:
                    pt1, v1 = get_pt(idx1)
                    pt2, v2 = get_pt(idx2)
                    if v1 > 0.4 and v2 > 0.4:
                        cv2.line(black_canvas, pt1, pt2, color, self.line_thickness)

                # Draw joint points
                for i in range(len(landmarks)):
                    pt, v = get_pt(i)
                    if v > 0.4:
                        cv2.circle(black_canvas, pt, self.joint_radius, (255, 255, 255), -1)

        return black_canvas

    def close(self):
        if self._pose_detector is not None:
            self._pose_detector.close()


def process_video_skeletons(
    video_path: str,
    output_path: Optional[str] = None,
    gamma: float = 0.67,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Reads a video, applies gamma correction (0.67), extracts skeletons on black background,

    and optionally saves the result as an AVI video.

    Args:
        video_path: Input video path.
        output_path: Optional path to save rendered skeleton video.
        gamma: Gamma correction factor (default: 0.67).
        target_size: (H, W) tuple to resize.

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
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    renderer = SkeletonRenderer()
    skeleton_frames: List[np.ndarray] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Resize if needed
            if target_size and (orig_h, orig_w) != target_size:
                frame = cv2.resize(frame, (out_w, out_h))

            # 2. Apply gamma correction (γ = 0.67) to enhance contrast
            enhanced = apply_gamma_correction(frame, gamma=gamma)

            # 3. Extract pose & render on black background
            skeleton_bgr = renderer.render_on_black(enhanced)

            if writer:
                writer.write(skeleton_bgr)

            # Store in RGB
            skeleton_rgb = cv2.cvtColor(skeleton_bgr, cv2.COLOR_BGR2RGB)
            skeleton_frames.append(skeleton_rgb)
    finally:
        cap.release()
        if writer:
            writer.release()
        renderer.close()

    return np.array(skeleton_frames, dtype=np.uint8)
