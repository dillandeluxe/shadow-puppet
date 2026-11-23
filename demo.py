"""Synthetic hand animation used when no camera is available."""

import math
from typing import List, Tuple

import numpy as np


from gestures import HandLandmarksData


# ------------------------------------------------------------------------------
# Synthetic Demonstration Generator (Headless / Camera-less Testing Fallback)
# ------------------------------------------------------------------------------
class SyntheticDemoGenerator:
    """
    Generates realistic animated hand landmark trajectories for testing and demonstration
    when a physical camera is unavailable or in automated test environments.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        self.t = 0.0

    def generate_frame_and_landmarks(self) -> Tuple[np.ndarray, List[HandLandmarksData]]:
        """
        Produces an animated procedural video frame and synthetic hand landmarks
        cycling through Bird, Dog, and Deer gestures smoothly.
        """
        self.t += 0.035
        cycle = int(self.t // 5) % 5  # Cycles through all 5 classic puppet shapes

        # Create aesthetic ambient canvas
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Gentle dark studio gradient
        for y in range(self.h):
            shade = int(25 + 20 * (y / self.h))
            frame[y, :] = (shade, shade + 5, shade + 8)

        cx = self.w * 0.5 + 40.0 * math.sin(self.t * 0.8)
        cy = self.h * 0.55 + 25.0 * math.cos(self.t * 1.1)
        scale = 140.0 + 20.0 * math.sin(self.t * 0.5)
        rot_deg = 15.0 * math.sin(self.t * 0.9)
        rad = math.radians(rot_deg)

        if cycle == 3:
            # Synthetic Two-Hand Bird: Crossed wrists/thumbs with flapping flight feathers
            wing_tilt = 20.0 * math.sin(self.t * 3.5)
            h1_rot = -25.0 - wing_tilt
            h2_rot = 25.0 + wing_tilt
            h1_rad = math.radians(h1_rot)
            h2_rad = math.radians(h2_rot)

            h1_px = np.zeros((21, 2), dtype=np.float32)
            h2_px = np.zeros((21, 2), dtype=np.float32)

            h1_px[0] = [cx - scale * 0.15, cy + scale * 0.2]
            h1_px[9] = [cx - scale * 0.35, cy - scale * 0.3]
            h1_px[4] = [cx, cy - scale * 0.1]
            for idx in [8, 12, 16, 20]:
                h1_px[idx] = [cx - scale * 0.8, cy - scale * 0.4]

            h2_px[0] = [cx + scale * 0.15, cy + scale * 0.2]
            h2_px[9] = [cx + scale * 0.35, cy - scale * 0.3]
            h2_px[4] = [cx, cy - scale * 0.1]
            for idx in [8, 12, 16, 20]:
                h2_px[idx] = [cx + scale * 0.8, cy - scale * 0.4]

            for i in range(21):
                if np.linalg.norm(h1_px[i]) < 1.0:
                    h1_px[i] = [cx - scale * 0.2, cy]
                if np.linalg.norm(h2_px[i]) < 1.0:
                    h2_px[i] = [cx + scale * 0.2, cy]

            fe_wings = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
            nd_wings = {
                "thumb_index": 0.6, "thumb_middle": 0.6, "thumb_ring": 0.6, "thumb_pinky": 0.6,
                "index_middle": 0.3, "middle_ring": 0.3, "ring_pinky": 0.3, "index_pinky": 0.9,
                "index_tip_mcp": 1.2, "middle_tip_mcp": 1.2, "ring_tip_mcp": 1.2, "pinky_tip_mcp": 1.2,
            }

            hand1 = HandLandmarksData(
                label="Left",
                raw_landmarks=[(p[0] / self.w, p[1] / self.h, 0.0) for p in h1_px],
                pixel_landmarks=h1_px,
                wrist_px=h1_px[0],
                middle_mcp_px=h1_px[9],
                palm_center_px=np.array([cx - scale * 0.2, cy], dtype=np.float32),
                hand_length_px=scale,
                axis_vector=np.array([math.sin(h1_rad), -math.cos(h1_rad)], dtype=np.float32),
                rotation_deg=h1_rot,
                finger_extended=fe_wings,
                norm_distances=nd_wings,
            )
            hand2 = HandLandmarksData(
                label="Right",
                raw_landmarks=[(p[0] / self.w, p[1] / self.h, 0.0) for p in h2_px],
                pixel_landmarks=h2_px,
                wrist_px=h2_px[0],
                middle_mcp_px=h2_px[9],
                palm_center_px=np.array([cx + scale * 0.2, cy], dtype=np.float32),
                hand_length_px=scale,
                axis_vector=np.array([math.sin(h2_rad), -math.cos(h2_rad)], dtype=np.float32),
                rotation_deg=h2_rot,
                finger_extended=fe_wings,
                norm_distances=nd_wings,
            )
            return frame, [hand1, hand2]

        # Build mock 21 landmarks for single-hand gestures
        px_pts = np.zeros((21, 2), dtype=np.float32)
        px_pts[0] = [cx, cy + scale * 0.5]  # Wrist
        px_pts[9] = [cx, cy - scale * 0.5]  # Middle MCP

        if cycle == 0:
            # Synthetic Deer: Index and Pinky extended, middle/ring touching thumb
            px_pts[8] = [cx - scale * 0.45, cy - scale * 1.2]
            px_pts[20] = [cx + scale * 0.45, cy - scale * 1.2]
            px_pts[4] = [cx, cy - scale * 0.7]
            px_pts[12] = [cx, cy - scale * 0.7]
            px_pts[16] = [cx, cy - scale * 0.7]
            fe = {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True}
        elif cycle == 1:
            # Synthetic Dog: Index up, middle & ring extended together, thumb barking
            px_pts[8] = [cx - scale * 0.35, cy - scale * 1.1]
            px_pts[12] = [cx + scale * 0.3, cy - scale * 0.9]
            px_pts[16] = [cx + scale * 0.28, cy - scale * 0.88]
            bark_y = cy - scale * (0.65 - 0.25 * abs(math.sin(self.t * 4.0)))
            px_pts[4] = [cx + scale * 0.25, bark_y]
            px_pts[20] = [cx - scale * 0.2, cy - scale * 0.4]
            fe = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": False}
        elif cycle == 2:
            # Synthetic Rabbit: Index & Middle up together
            px_pts[8] = [cx - scale * 0.1, cy - scale * 1.2]
            px_pts[12] = [cx + scale * 0.1, cy - scale * 1.2]
            px_pts[4] = [cx, cy - scale * 0.5]
            px_pts[16] = [cx + scale * 0.1, cy - scale * 0.3]
            px_pts[20] = [cx + scale * 0.2, cy - scale * 0.3]
            fe = {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}
        else:
            # Synthetic Swan: Vertical neck with pinched beak
            rot_deg = 5.0 * math.sin(self.t * 0.8)
            rad = math.radians(rot_deg)
            beak_u = cx + scale * 0.25
            beak_v = cy - scale * 1.15
            px_pts[4] = [beak_u, beak_v]
            px_pts[8] = [beak_u + 3, beak_v]
            px_pts[12] = [beak_u + 2, beak_v + 3]
            px_pts[16] = [cx - scale * 0.15, cy - scale * 0.5]
            px_pts[20] = [cx - scale * 0.25, cy - scale * 0.4]
            fe = {"thumb": True, "index": True, "middle": True, "ring": False, "pinky": False}

        # Fill remaining joints
        for i in range(21):
            if np.linalg.norm(px_pts[i]) < 1.0:
                px_pts[i] = [cx, cy]

        norm_dists = {
            "thumb_index": float(np.linalg.norm(px_pts[4] - px_pts[8]) / scale),
            "thumb_middle": float(np.linalg.norm(px_pts[4] - px_pts[12]) / scale),
            "thumb_ring": float(np.linalg.norm(px_pts[4] - px_pts[16]) / scale),
            "thumb_pinky": float(np.linalg.norm(px_pts[4] - px_pts[20]) / scale),
            "index_middle": float(np.linalg.norm(px_pts[8] - px_pts[12]) / scale),
            "middle_ring": float(np.linalg.norm(px_pts[12] - px_pts[16]) / scale),
            "ring_pinky": float(np.linalg.norm(px_pts[16] - px_pts[20]) / scale),
            "index_pinky": float(np.linalg.norm(px_pts[8] - px_pts[20]) / scale),
            "index_tip_mcp": 1.2,
            "middle_tip_mcp": 1.2 if fe["middle"] else 0.4,
            "ring_tip_mcp": 1.2 if fe["ring"] else 0.4,
            "pinky_tip_mcp": 1.2 if fe["pinky"] else 0.4,
        }

        mock_hand = HandLandmarksData(
            label="Right",
            raw_landmarks=[(p[0] / self.w, p[1] / self.h, 0.0) for p in px_pts],
            pixel_landmarks=px_pts,
            wrist_px=px_pts[0],
            middle_mcp_px=px_pts[9],
            palm_center_px=np.array([cx, cy], dtype=np.float32),
            hand_length_px=scale,
            axis_vector=np.array([math.sin(rad), -math.cos(rad)], dtype=np.float32),
            rotation_deg=rot_deg,
            finger_extended=fe,
            norm_distances=norm_dists,
        )

        return frame, [mock_hand]
