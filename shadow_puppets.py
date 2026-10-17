#!/usr/bin/env python3
"""
================================================================================
Hand-Gesture Shadow Puppets AI
================================================================================
A real-time Computer Vision & Interactive Graphics application that captures a
live webcam feed, tracks hand landmarks using Google MediaPipe Hands, evaluates
geometric heuristics to recognize classic shadow puppet shapes (Bird, Dog,
Deer/Goat, Rabbit, Swan), and dynamically renders smoothly-tracked, stylized 2D
silhouette puppets directly over the frame with authentic shadow theatre aesthetics.

Key Features:
- 30+ FPS Real-time Hand Tracking using MediaPipe Hands (max_num_hands=2).
- Zero Loose Pixel-Matching: Pure scale-invariant geometric heuristics based on
  normalized Euclidean distances, joint-vector projections, and finger extension states.
- 5 Distinct Classic Shadow Puppets:
    1. Bird / Eagle (Two-handed crossed wrists/thumbs with flapping wings, or one-handed dove)
    2. Dog / Wolf (Erect ears, extended snout, and real-time openable jaw via thumb)
    3. Deer / Goat (High index/pinky antler horns, middle/ring touching thumb muzzle)
    4. Rabbit / Bunny (Close parallel ears, curled paws, twitchy nose)
    5. Swan / Goose (Arched S-neck forearm, pinched finger-thumb beak)
- Real-Time Dynamic Kinematics:
    * Translation: Centroid / anchor tracking.
    * Scaling: Scale-normalized hand span from wrist to middle MCP joint.
    * Rotation: Smooth hand-axis angular tracking with atan2 vector orientation.
    * Expressive Kinematics: Real-time jaw opening for the dog and wing flapping for the bird.
- High-Performance Procedural Rendering:
    * 2D Affine Polygon Transformation using lightweight NumPy matrix operations.
    * Anti-aliased silhouette rendering with warm translucent shadow theater tones.
    * Glowing eye cutouts and authentic ambient parchment backlight mode ('s' key).
- Temporal State Machine & Anti-Jitter:
    * Exponential Moving Average (EMA) position, scale, and angular smoothing.
    * Gesture state debouncing & hysteresis to prevent violent snapping.
    * Cross-fade alpha blending for cinematic puppet morphing.
- Interactive HUD & Controls:
    * High-contrast glassmorphism status pill displaying active gesture & confidence.
    * Live FPS monitor, landmark skeleton toggle ('l'), shadow screen mode ('s'),
      camera flip ('c'), and on-screen gesture cheat-sheet ('h').
    * Headless / Demo Fallback mode ('--demo') for testing without physical webcam.

Dependencies:
    pip install opencv-python mediapipe numpy

Usage:
    python shadow_puppets.py
    python shadow_puppets.py --camera 0
    python shadow_puppets.py --demo
================================================================================
"""

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ShadowPuppetsAI")


# ------------------------------------------------------------------------------
# Gesture Classification Enums and Data Models
# ------------------------------------------------------------------------------
class PuppetGesture(str, Enum):
    """Supported shadow puppet gesture classes."""
    NONE = "None"
    BIRD = "Bird / Eagle"
    DOG = "Dog / Wolf"
    DEER = "Deer / Goat"
    RABBIT = "Rabbit / Bunny"
    SWAN = "Swan / Goose"


@dataclass
class HandLandmarksData:
    """Pre-processed geometric features extracted from raw 21 MediaPipe landmarks."""
    label: str  # "Left" or "Right"
    raw_landmarks: List[Tuple[float, float, float]]  # (x, y, z) normalized [0, 1]
    pixel_landmarks: np.ndarray  # (21, 2) absolute pixel coordinates (x, y)
    wrist_px: np.ndarray  # (2,) [x, y]
    middle_mcp_px: np.ndarray  # (2,) [x, y]
    palm_center_px: np.ndarray  # (2,) [x, y]
    hand_length_px: float  # Euclidean distance between Wrist and Middle MCP
    axis_vector: np.ndarray  # Normalized unit vector along hand longitudinal axis
    rotation_deg: float  # Angle relative to vertical in screen coordinates
    finger_extended: Dict[str, bool]  # 'thumb', 'index', 'middle', 'ring', 'pinky'
    norm_distances: Dict[str, float]  # Normalized Euclidean distances by hand_length_px


@dataclass
class PuppetPose:
    """Target transform and kinematic parameters for puppet rendering."""
    gesture: PuppetGesture = PuppetGesture.NONE
    confidence: float = 0.0
    center_px: Tuple[float, float] = (0.0, 0.0)
    scale_factor: float = 1.0
    rotation_deg: float = 0.0
    flip_horizontal: bool = False
    # Dynamic kinematic morphing parameters
    jaw_open: float = 0.0  # 0.0 (closed) to 1.0 (fully open) for Dog
    left_wing_tilt: float = 0.0  # Wing flap angle for Bird
    right_wing_tilt: float = 0.0  # Wing flap angle for Bird
    extra_data: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Landmark Extraction & Normalization
# ------------------------------------------------------------------------------
class HandFeatureExtractor:
    """
    Extracts scale-invariant 2D/3D metrics from raw MediaPipe hand landmarks.
    Computes finger extension statuses, anchor positions, and longitudinal orientation.
    """

    # MediaPipe Hand Landmark Indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    @classmethod
    def process_hand(
        cls,
        landmarks: Any,
        handedness_label: str,
        frame_width: int,
        frame_height: int,
    ) -> HandLandmarksData:
        """
        Converts MediaPipe landmarks into a structured HandLandmarksData container
        with normalized geometric metrics.
        """
        lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        raw_pts = [(lm.x, lm.y, lm.z) for lm in lm_list]
        px_pts = np.array(
            [[lm.x * frame_width, lm.y * frame_height] for lm in lm_list],
            dtype=np.float32,
        )

        wrist = px_pts[cls.WRIST]
        mid_mcp = px_pts[cls.MIDDLE_MCP]
        index_mcp = px_pts[cls.INDEX_MCP]
        pinky_mcp = px_pts[cls.PINKY_MCP]

        # Hand longitudinal vector and base hand scale
        hand_vec = mid_mcp - wrist
        hand_len = float(np.linalg.norm(hand_vec))
        if hand_len < 1e-4:
            hand_len = 1.0

        unit_axis = hand_vec / hand_len

        # Palm center: average of Wrist, Index MCP, Middle MCP, Pinky MCP
        palm_center = (wrist + index_mcp + mid_mcp + pinky_mcp) / 4.0

        # Hand rotation angle in screen space:
        # In screen space, -Y is UP. A vertical upright hand pointing up has dy < 0.
        # atan2(dy, dx) + pi/2 gives 0 deg when pointing vertically upward.
        angle_rad = math.atan2(float(hand_vec[1]), float(hand_vec[0])) + (math.pi / 2.0)
        angle_deg = math.degrees(angle_rad)

        # Helper Euclidean distance normalized by base hand length
        def n_dist(idx1: int, idx2: int) -> float:
            return float(np.linalg.norm(px_pts[idx1] - px_pts[idx2]) / hand_len)

        # Finger Extension Logic:
        # For fingers (Index, Middle, Ring, Pinky), test whether the TIP is further from
        # the MCP/WRIST than the PIP joint along the finger projection axis.
        def is_finger_extended(mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int) -> bool:
            dist_wrist_tip = float(np.linalg.norm(px_pts[tip_idx] - wrist))
            dist_wrist_pip = float(np.linalg.norm(px_pts[pip_idx] - wrist))
            dist_mcp_tip = float(np.linalg.norm(px_pts[tip_idx] - px_pts[mcp_idx]))
            dist_mcp_pip = float(np.linalg.norm(px_pts[pip_idx] - px_pts[mcp_idx]))
            return (dist_wrist_tip > dist_wrist_pip * 1.1) and (dist_mcp_tip > dist_mcp_pip * 1.2)

        # Thumb Extension:
        # Thumb operates on an oblique plane; compare distance of thumb tip to pinky MCP vs IP to pinky MCP
        thumb_tip_pinky = float(np.linalg.norm(px_pts[cls.THUMB_TIP] - pinky_mcp))
        thumb_ip_pinky = float(np.linalg.norm(px_pts[cls.THUMB_IP] - pinky_mcp))
        thumb_extended = thumb_tip_pinky > thumb_ip_pinky * 1.15 and (n_dist(cls.THUMB_TIP, cls.INDEX_MCP) > 0.45)

        finger_extended = {
            "thumb": thumb_extended,
            "index": is_finger_extended(cls.INDEX_MCP, cls.INDEX_PIP, cls.INDEX_DIP, cls.INDEX_TIP),
            "middle": is_finger_extended(cls.MIDDLE_MCP, cls.MIDDLE_PIP, cls.MIDDLE_DIP, cls.MIDDLE_TIP),
            "ring": is_finger_extended(cls.RING_MCP, cls.RING_PIP, cls.RING_DIP, cls.RING_TIP),
            "pinky": is_finger_extended(cls.PINKY_MCP, cls.PINKY_PIP, cls.PINKY_DIP, cls.PINKY_TIP),
        }

        # Cached normalized pairwise distances frequently referenced by heuristics
        norm_distances = {
            "thumb_index": n_dist(cls.THUMB_TIP, cls.INDEX_TIP),
            "thumb_middle": n_dist(cls.THUMB_TIP, cls.MIDDLE_TIP),
            "thumb_ring": n_dist(cls.THUMB_TIP, cls.RING_TIP),
            "thumb_pinky": n_dist(cls.THUMB_TIP, cls.PINKY_TIP),
            "index_middle": n_dist(cls.INDEX_TIP, cls.MIDDLE_TIP),
            "middle_ring": n_dist(cls.MIDDLE_TIP, cls.RING_TIP),
            "ring_pinky": n_dist(cls.RING_TIP, cls.PINKY_TIP),
            "index_pinky": n_dist(cls.INDEX_TIP, cls.PINKY_TIP),
            "index_tip_mcp": n_dist(cls.INDEX_TIP, cls.INDEX_MCP),
            "middle_tip_mcp": n_dist(cls.MIDDLE_TIP, cls.MIDDLE_MCP),
            "ring_tip_mcp": n_dist(cls.RING_TIP, cls.RING_MCP),
            "pinky_tip_mcp": n_dist(cls.PINKY_TIP, cls.PINKY_MCP),
        }

        return HandLandmarksData(
            label=handedness_label,
            raw_landmarks=raw_pts,
            pixel_landmarks=px_pts,
            wrist_px=wrist,
            middle_mcp_px=mid_mcp,
            palm_center_px=palm_center,
            hand_length_px=hand_len,
            axis_vector=unit_axis,
            rotation_deg=angle_deg,
            finger_extended=finger_extended,
            norm_distances=norm_distances,
        )


# ------------------------------------------------------------------------------
# Geometric Heuristic Gesture Recognizer
# ------------------------------------------------------------------------------
class GestureRecognizer:
    """
    Evaluates strictly geometric, scale-invariant heuristics to identify puppet
    postures with zero reliance on arbitrary pixel classification.
    """

    @classmethod
    def evaluate(cls, hands_data: List[HandLandmarksData]) -> PuppetPose:
        """
        Evaluates active hand gestures from 1 or 2 hands and returns the top PuppetPose.
        Prioritizes two-hand shapes (Bird) when two hands are present, falling back to
        single-hand shapes (Dog, Deer, Rabbit, Swan, Single Bird).
        """
        if not hands_data:
            return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

        # ----------------------------------------------------------------------
        # Check Two-Hand Bird (Classic Shadow Theatre)
        # ----------------------------------------------------------------------
        if len(hands_data) >= 2:
            bird_pose = cls._check_two_hand_bird(hands_data[0], hands_data[1])
            if bird_pose.confidence >= 0.70:
                return bird_pose

        # ----------------------------------------------------------------------
        # Evaluate Single-Hand Gestures across available hands
        # ----------------------------------------------------------------------
        best_pose = PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

        for hand in hands_data:
            # Check Deer / Goat
            deer_pose = cls._check_deer(hand)
            if deer_pose.confidence > best_pose.confidence:
                best_pose = deer_pose

            # Check Dog / Wolf
            dog_pose = cls._check_dog(hand)
            if dog_pose.confidence > best_pose.confidence:
                best_pose = dog_pose

            # Check Rabbit / Bunny
            rabbit_pose = cls._check_rabbit(hand)
            if rabbit_pose.confidence > best_pose.confidence:
                best_pose = rabbit_pose

            # Check Swan / Goose
            swan_pose = cls._check_swan(hand)
            if swan_pose.confidence > best_pose.confidence:
                best_pose = swan_pose

            # Check Single-Hand Bird
            single_bird_pose = cls._check_single_hand_bird(hand)
            if single_bird_pose.confidence > best_pose.confidence:
                best_pose = single_bird_pose

        return best_pose

    @classmethod
    def _check_two_hand_bird(
        cls, hand1: HandLandmarksData, hand2: HandLandmarksData
    ) -> PuppetPose:
        """
        Two-Hand Bird Heuristic:
        In classical shadow puppetry, wrists or thumbs cross/overlap while both hands
        splay fingers outwards to emulate flapping wings.
        """
        avg_scale = (hand1.hand_length_px + hand2.hand_length_px) / 2.0

        # Distance between wrists and between thumbs
        wrist_dist = np.linalg.norm(hand1.wrist_px - hand2.wrist_px) / avg_scale
        thumb1_tip = hand1.pixel_landmarks[HandFeatureExtractor.THUMB_TIP]
        thumb2_tip = hand2.pixel_landmarks[HandFeatureExtractor.THUMB_TIP]
        thumb_dist = np.linalg.norm(thumb1_tip - thumb2_tip) / avg_scale

        # Thumbs or wrists must be in close proximity
        base_connected = (wrist_dist < 1.4) or (thumb_dist < 0.9)

        # Wings: fingers spread outward
        h1_ext_count = sum(1 for f in ["index", "middle", "ring", "pinky"] if hand1.finger_extended[f])
        h2_ext_count = sum(1 for f in ["index", "middle", "ring", "pinky"] if hand2.finger_extended[f])

        # Hands should point towards opposing lateral directions
        h1_dir_x = hand1.axis_vector[0]
        h2_dir_x = hand2.axis_vector[0]
        opposing_spread = (h1_dir_x * h2_dir_x < 0.2)  # Lateral divergence

        if base_connected and (h1_ext_count >= 3) and (h2_ext_count >= 3):
            # Confidence score calculation
            conf = 0.70
            if wrist_dist < 0.9 or thumb_dist < 0.6:
                conf += 0.15
            if opposing_spread:
                conf += 0.10
            if h1_ext_count == 4 and h2_ext_count == 4:
                conf += 0.05
            conf = min(conf, 0.98)

            center_px = tuple(((hand1.palm_center_px + hand2.palm_center_px) / 2.0).tolist())

            # Wing tilt angles from vertical for dynamic flapping animation
            # Determine which hand is left and which is right on screen
            if hand1.palm_center_px[0] < hand2.palm_center_px[0]:
                left_h, right_h = hand1, hand2
            else:
                left_h, right_h = hand2, hand1

            left_tilt = -left_h.rotation_deg * 0.5
            right_tilt = right_h.rotation_deg * 0.5

            return PuppetPose(
                gesture=PuppetGesture.BIRD,
                confidence=conf,
                center_px=center_px,
                scale_factor=avg_scale,
                rotation_deg=0.0,  # Two-hand bird maintains stabilized body axis
                flip_horizontal=False,
                left_wing_tilt=left_tilt,
                right_wing_tilt=right_tilt,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

    @classmethod
    def _check_deer(cls, hand: HandLandmarksData) -> PuppetPose:
        """
        Deer / Goat Heuristic:
        - Index and Pinky extended (antlers/horns).
        - Middle and Ring bent/curled down.
        - Middle & Ring fingertips in close contact with Thumb tip (forming muzzle/mouth).
        - Wide antler spread between Index and Pinky.
        """
        fe = hand.finger_extended
        nd = hand.norm_distances

        # Core requirement: Index & Pinky UP; Middle & Ring DOWN
        antlers_up = fe["index"] and fe["pinky"]
        muzzle_down = (not fe["middle"]) and (not fe["ring"])

        if not (antlers_up and muzzle_down):
            return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

        # Contact: Middle & Ring tips touching or touching Thumb tip
        touch_thumb_middle = nd["thumb_middle"] < 0.52
        touch_thumb_ring = nd["thumb_ring"] < 0.52
        touch_middle_ring = nd["middle_ring"] < 0.35
        antler_spread = nd["index_pinky"] > 0.65

        if (touch_thumb_middle or touch_thumb_ring or touch_middle_ring) and antler_spread:
            conf = 0.75
            if touch_thumb_middle and touch_thumb_ring:
                conf += 0.12
            if nd["index_pinky"] > 0.85:
                conf += 0.08
            conf = min(conf, 0.98)

            # Anchor: Palm center or midpoint between wrist and middle MCP
            center = tuple(hand.palm_center_px.tolist())
            flip = hand.label == "Left"

            return PuppetPose(
                gesture=PuppetGesture.DEER,
                confidence=conf,
                center_px=center,
                scale_factor=hand.hand_length_px,
                rotation_deg=hand.rotation_deg,
                flip_horizontal=flip,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

    @classmethod
    def _check_dog(cls, hand: HandLandmarksData) -> PuppetPose:
        """
        Dog / Barking Wolf Heuristic:
        - Index finger raised/extended (erect ear).
        - Middle & Ring fingers extended straight together (snout/upper muzzle).
        - Pinky can be curled or relaxed.
        - Thumb positioned underneath forming the lower jaw.
        - Real-time kinematic parameter: jaw_open derived from thumb-to-snout distance!
        """
        fe = hand.finger_extended
        nd = hand.norm_distances

        # Ear up (Index extended); Snout forward (Middle and/or Ring extended)
        ear_up = fe["index"]
        snout_together = nd["middle_ring"] < 0.38
        middle_extended = fe["middle"]

        if ear_up and middle_extended and snout_together:
            # Thumb creates the lower jaw
            # Measure distance from thumb tip to middle tip
            thumb_mid_dist = nd["thumb_middle"]

            # Dog jaw openness normalized [0.0 = closed, 1.0 = fully open barking]
            # Normal closed jaw: thumb touches ring/middle (~0.25). Open jaw: thumb pulls away (>0.60)
            jaw_open = float(np.clip((thumb_mid_dist - 0.25) / 0.40, 0.0, 1.0))

            conf = 0.72
            if not fe["pinky"]:
                conf += 0.10  # Curled pinky makes a cleaner silhouette
            if nd["middle_ring"] < 0.25:
                conf += 0.10
            conf = min(conf, 0.96)

            center = tuple(hand.palm_center_px.tolist())
            flip = hand.label == "Left"

            return PuppetPose(
                gesture=PuppetGesture.DOG,
                confidence=conf,
                center_px=center,
                scale_factor=hand.hand_length_px,
                rotation_deg=hand.rotation_deg,
                flip_horizontal=flip,
                jaw_open=jaw_open,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

    @classmethod
    def _check_rabbit(cls, hand: HandLandmarksData) -> PuppetPose:
        """
        Rabbit / Bunny Heuristic:
        - Index and Middle extended upright parallel to each other (the two bunny ears).
        - Ring and Pinky curled tightly into palm.
        - Thumb folded over ring finger.
        - Small ear gap between index and middle tips.
        """
        fe = hand.finger_extended
        nd = hand.norm_distances

        ears_up = fe["index"] and fe["middle"]
        paws_curled = (not fe["ring"]) and (not fe["pinky"])
        ears_parallel = nd["index_middle"] < 0.42

        if ears_up and paws_curled and ears_parallel:
            conf = 0.75
            if nd["ring_tip_mcp"] < 0.55 and nd["pinky_tip_mcp"] < 0.55:
                conf += 0.12
            if nd["index_middle"] < 0.30:
                conf += 0.08
            conf = min(conf, 0.98)

            center = tuple(hand.palm_center_px.tolist())
            flip = hand.label == "Left"

            return PuppetPose(
                gesture=PuppetGesture.RABBIT,
                confidence=conf,
                center_px=center,
                scale_factor=hand.hand_length_px,
                rotation_deg=hand.rotation_deg,
                flip_horizontal=flip,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

    @classmethod
    def _check_swan(cls, hand: HandLandmarksData) -> PuppetPose:
        """
        Swan / Goose Heuristic:
        - Hand held vertically with forearm acting as the neck.
        - Thumb, Index, and Middle tips pinched together forming the beak.
        - Ring and Pinky relaxed or curved behind head.
        """
        nd = hand.norm_distances

        # Pinch between Thumb, Index, and Middle
        beak_pinch = (nd["thumb_index"] < 0.32) and (nd["thumb_middle"] < 0.35)
        snout_tight = nd["index_middle"] < 0.28

        if beak_pinch and snout_tight:
            conf = 0.72
            if nd["thumb_index"] < 0.20:
                conf += 0.12
            conf = min(conf, 0.95)

            center = tuple(hand.palm_center_px.tolist())
            flip = hand.label == "Left"

            return PuppetPose(
                gesture=PuppetGesture.SWAN,
                confidence=conf,
                center_px=center,
                scale_factor=hand.hand_length_px,
                rotation_deg=hand.rotation_deg,
                flip_horizontal=flip,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)

    @classmethod
    def _check_single_hand_bird(cls, hand: HandLandmarksData) -> PuppetPose:
        """
        Single-Hand Bird / Flying Dove Heuristic:
        - Thumb extended upwards/forward as the bird's head and neck.
        - Index, Middle, Ring, Pinky extended and spread like plumage/wings.
        """
        fe = hand.finger_extended
        nd = hand.norm_distances

        wings_open = fe["index"] and fe["middle"] and fe["ring"] and fe["pinky"]
        thumb_head = fe["thumb"] and (nd["thumb_index"] > 0.60)

        if wings_open and thumb_head:
            conf = 0.73
            if nd["index_pinky"] > 0.90:
                conf += 0.12
            conf = min(conf, 0.92)

            center = tuple(hand.palm_center_px.tolist())
            flip = hand.label == "Left"

            return PuppetPose(
                gesture=PuppetGesture.BIRD,
                confidence=conf,
                center_px=center,
                scale_factor=hand.hand_length_px,
                rotation_deg=hand.rotation_deg,
                flip_horizontal=flip,
                left_wing_tilt=0.0,
                right_wing_tilt=0.0,
            )

        return PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)


# ------------------------------------------------------------------------------
# Temporal Smoothing & State Machine
# ------------------------------------------------------------------------------
class TemporalSmoother:
    """
    Prevents visual jitter, popping, and violent snapping:
    - Low-pass Exponential Moving Average (EMA) for translations, scale, and rotations.
    - Continuous angular unwrapping to prevent 360-degree boundary jumps.
    - Gesture state debouncing hysteresis (candidate confirmation frames).
    - Smooth cross-fade alpha blending during gesture transitions.
    """

    def __init__(
        self,
        alpha_pos: float = 0.35,
        alpha_scale: float = 0.30,
        alpha_rot: float = 0.30,
        fade_frames: int = 12,
        debounce_frames: int = 3,
    ):
        self.alpha_pos = alpha_pos
        self.alpha_scale = alpha_scale
        self.alpha_rot = alpha_rot
        self.fade_frames = fade_frames
        self.debounce_frames = debounce_frames

        # State tracking
        self.current_gesture = PuppetGesture.NONE
        self.candidate_gesture = PuppetGesture.NONE
        self.candidate_count = 0

        # Alpha cross-fade
        self.active_alpha = 0.0  # 0.0 to 1.0

        # Filtered kinematics
        self.smoothed_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.smoothed_scale = 100.0
        self.smoothed_cos = 1.0
        self.smoothed_sin = 0.0
        self.smoothed_jaw = 0.0
        self.smoothed_left_wing = 0.0
        self.smoothed_right_wing = 0.0
        self.flip_horizontal = False
        self.confidence = 0.0

    def update(self, raw_pose: PuppetPose) -> Tuple[PuppetPose, float]:
        """
        Updates the internal state machine and returns the smoothed pose and current alpha.
        """
        # 1. State Hysteresis & Debouncing
        incoming_gesture = raw_pose.gesture

        if incoming_gesture == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = incoming_gesture
            self.candidate_count = 1

        if self.candidate_count >= self.debounce_frames:
            self.current_gesture = self.candidate_gesture

        # 2. Alpha fade-in / fade-out
        if self.current_gesture != PuppetGesture.NONE:
            self.active_alpha = min(1.0, self.active_alpha + (1.0 / self.fade_frames))
        else:
            self.active_alpha = max(0.0, self.active_alpha - (1.0 / self.fade_frames))

        # 3. Low-Pass Kinematic Smoothing
        raw_pos = np.array(raw_pose.center_px, dtype=np.float32)
        if np.linalg.norm(self.smoothed_pos) < 1.0:
            self.smoothed_pos = raw_pos
            self.smoothed_scale = raw_pose.scale_factor
        else:
            self.smoothed_pos = self.alpha_pos * raw_pos + (1.0 - self.alpha_pos) * self.smoothed_pos
            self.smoothed_scale = (
                self.alpha_scale * raw_pose.scale_factor + (1.0 - self.alpha_scale) * self.smoothed_scale
            )

        # Angular smoothing via unit circle vector components to avoid wraparound jumps
        rad = math.radians(raw_pose.rotation_deg)
        curr_cos = math.cos(rad)
        curr_sin = math.sin(rad)
        self.smoothed_cos = self.alpha_rot * curr_cos + (1.0 - self.alpha_rot) * self.smoothed_cos
        self.smoothed_sin = self.alpha_rot * curr_sin + (1.0 - self.alpha_rot) * self.smoothed_sin
        smoothed_angle_deg = math.degrees(math.atan2(self.smoothed_sin, self.smoothed_cos))

        # Dynamic Kinematic Parameters
        self.smoothed_jaw = 0.4 * raw_pose.jaw_open + 0.6 * self.smoothed_jaw
        self.smoothed_left_wing = 0.4 * raw_pose.left_wing_tilt + 0.6 * self.smoothed_left_wing
        self.smoothed_right_wing = 0.4 * raw_pose.right_wing_tilt + 0.6 * self.smoothed_right_wing
        self.flip_horizontal = raw_pose.flip_horizontal
        self.confidence = raw_pose.confidence

        smoothed_pose = PuppetPose(
            gesture=self.current_gesture,
            confidence=self.confidence,
            center_px=(float(self.smoothed_pos[0]), float(self.smoothed_pos[1])),
            scale_factor=float(self.smoothed_scale),
            rotation_deg=float(smoothed_angle_deg),
            flip_horizontal=self.flip_horizontal,
            jaw_open=float(self.smoothed_jaw),
            left_wing_tilt=float(self.smoothed_left_wing),
            right_wing_tilt=float(self.smoothed_right_wing),
        )

        return smoothed_pose, self.active_alpha


# ------------------------------------------------------------------------------
# Procedural 2D Silhouette Puppet Renderer
# ------------------------------------------------------------------------------
class PuppetRenderer:
    """
    Renders clean, stylized 2D procedural vector silhouettes directly onto video frames.
    Transforms canonical contour coordinates via lightweight 2D affine matrix math.
    Applies authentic translucent shadow theater styling with glowing cutout eyes.
    """

    # Authentic Shadow Theater Color Palette (BGR)
    CHARCOAL_SHADOW = (26, 24, 24)       # Deep leather shadow puppet tone
    AMBER_RIM_GLOW = (45, 120, 215)      # Warm parchment light diffraction edge
    EYE_APERTURE_COLOR = (210, 235, 250) # Translucent glowing aperture
    PARCHMENT_TINT = (205, 230, 242)     # Backlit parchment paper ambiance

    @classmethod
    def render(
        cls,
        frame: np.ndarray,
        pose: PuppetPose,
        alpha: float,
        shadow_screen_mode: bool = False,
    ) -> np.ndarray:
        """
        Renders the active puppet silhouette over the target frame.
        """
        if alpha <= 0.01 or pose.gesture == PuppetGesture.NONE:
            if shadow_screen_mode:
                return cls._apply_parchment_screen(frame)
            return frame

        # If Shadow Screen Mode is enabled, transform frame into parchment canvas
        output = cls._apply_parchment_screen(frame) if shadow_screen_mode else frame.copy()

        # Generate canonical 2D vector polygons for the active gesture
        polygons, eye_circles = cls._get_canonical_geometry(pose)
        if not polygons:
            return output

        # Construct 2D Affine Transformation Matrix:
        # Scale: normalized to reference length of 100 units
        s = (pose.scale_factor / 95.0) * 1.35
        rad = math.radians(pose.rotation_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Horizontal flip if user is using opposing hand
        fx = -1.0 if pose.flip_horizontal else 1.0

        # Affine 2x2 rotation-scaling matrix
        # P_screen = Anchor + s * R * [fx * u, v]^T
        M = np.array(
            [
                [s * cos_a * fx, -s * sin_a],
                [s * sin_a * fx,  s * cos_a],
            ],
            dtype=np.float32,
        )
        anchor = np.array(pose.center_px, dtype=np.float32)

        # Create overlay layer for smooth anti-aliased alpha blending
        h, w = output.shape[:2]
        overlay = np.zeros_like(output)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Transform and draw each polygon component
        for poly in polygons:
            poly_arr = np.array(poly, dtype=np.float32)
            transformed = (poly_arr @ M.T) + anchor
            pts_int = np.int32(transformed)

            # Fill silhouette polygon
            cv2.fillPoly(overlay, [pts_int], cls.CHARCOAL_SHADOW, lineType=cv2.LINE_AA)
            cv2.fillPoly(mask, [pts_int], 255, lineType=cv2.LINE_AA)

            # Soft rim-glow edge
            cv2.polylines(overlay, [pts_int], isClosed=True, color=cls.AMBER_RIM_GLOW, thickness=2, lineType=cv2.LINE_AA)

        # Transform and carve glowing eye apertures
        for center_u, center_v, radius in eye_circles:
            eye_pos = (np.array([center_u, center_v], dtype=np.float32) @ M.T) + anchor
            eye_r = max(2, int(radius * s))
            pt_center = (int(eye_pos[0]), int(eye_pos[1]))

            # Draw glowing aperture
            cv2.circle(overlay, pt_center, eye_r, cls.EYE_APERTURE_COLOR, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, pt_center, eye_r + 1, cls.AMBER_RIM_GLOW, thickness=1, lineType=cv2.LINE_AA)

        # Alpha composite overlay over frame
        effective_alpha = float(np.clip(alpha * 0.88, 0.0, 1.0))
        mask_3ch = cv2.merge([mask, mask, mask]) > 0
        blended = cv2.addWeighted(overlay, effective_alpha, output, 1.0 - effective_alpha * 0.4, 0)
        np.copyto(output, blended, where=mask_3ch)

        return output

    @classmethod
    def _apply_parchment_screen(cls, frame: np.ndarray) -> np.ndarray:
        """
        Simulates an authentic warm backlit parchment screen, casting dramatic hand shadows.
        """
        # Convert to grayscale to preserve hand shadow contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Invert slightly and apply vignette / warm backlight
        parchment = np.zeros_like(frame)
        parchment[:, :, 0] = np.uint8(np.clip(gray * 0.70 + 35, 0, 255))  # B
        parchment[:, :, 1] = np.uint8(np.clip(gray * 0.85 + 45, 0, 255))  # G
        parchment[:, :, 2] = np.uint8(np.clip(gray * 0.95 + 50, 0, 255))  # R
        return cv2.addWeighted(parchment, 0.85, frame, 0.15, 0)

    @classmethod
    def _get_canonical_geometry(
        cls, pose: PuppetPose
    ) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float, float]]]:
        """
        Returns procedural canonical 2D contours and eye apertures for each puppet.
        Coordinates are defined around (0, 0) anchor:
        +v is along the fingers/hand axis, +u is lateral.
        """
        polygons: List[List[Tuple[float, float]]] = []
        eyes: List[Tuple[float, float, float]] = []

        if pose.gesture == PuppetGesture.DOG:
            # ------------------------------------------------------------------
            # Stylized Dog / Barking Wolf Silhouette
            # ------------------------------------------------------------------
            jaw_angle = math.radians(pose.jaw_open * 22.0)  # Dynamic jaw rotation
            cos_j = math.cos(jaw_angle)
            sin_j = math.sin(jaw_angle)

            # Upper Head, Ears, and Upper Snout
            head_pts = [
                # Neck base
                (-25.0, -15.0),
                (-35.0, 30.0),
                # Back of head
                (-32.0, 75.0),
                # Erect triangular ear
                (-28.0, 105.0),
                (-18.0, 145.0),  # Ear tip
                (-2.0, 110.0),   # Inner ear base
                # Forehead / brow
                (5.0, 100.0),
                # Bridge of muzzle
                (18.0, 125.0),
                (35.0, 135.0),   # Nose tip
                # Upper mouth shelf
                (32.0, 115.0),
                (20.0, 95.0),
                # Jaw hinge connection
                (10.0, 80.0),
                # Throat
                (-5.0, 45.0),
                (-15.0, -15.0),
            ]
            polygons.append(head_pts)

            # Dynamic Lower Jaw Polygon (Rotates with jaw_open factor!)
            # Pivots around hinge (10.0, 80.0)
            hinge_u, hinge_v = 10.0, 80.0
            raw_jaw = [
                (10.0, 80.0),
                (22.0, 92.0),
                (32.0, 108.0),  # Lower chin tip
                (22.0, 100.0),
                (12.0, 72.0),
                (5.0, 50.0),
            ]
            rotated_jaw = []
            for u, v in raw_jaw:
                du = u - hinge_u
                dv = v - hinge_v
                # Rotate downward in lateral space
                ru = hinge_u + (du * cos_j + dv * sin_j)
                rv = hinge_v + (-du * sin_j + dv * cos_j)
                rotated_jaw.append((ru, rv))

            polygons.append(rotated_jaw)

            # Eye aperture (almond cutout)
            eyes.append((-5.0, 95.0, 4.5))

        elif pose.gesture == PuppetGesture.DEER:
            # ------------------------------------------------------------------
            # Stylized Deer / Regal Stag with Branching Antlers
            # ------------------------------------------------------------------
            # Head, Snout, Neck
            head_pts = [
                (-18.0, -20.0), # Neck base
                (-25.0, 40.0),
                (-20.0, 75.0),  # Jaw
                (-8.0, 110.0),  # Slender muzzle tip
                (5.0, 105.0),
                (10.0, 85.0),   # Forehead / brow
                # Antler pedicle
                (15.0, 88.0),
                # Ear base
                (20.0, 78.0),
                (35.0, 90.0),   # Leaf ear tip
                (22.0, 68.0),
                # Back of neck
                (15.0, 30.0),
                (10.0, -20.0),
            ]
            polygons.append(head_pts)

            # Branching Antler Rack (Majestic procedural tines)
            antlers = [
                (15.0, 88.0),   # Antler base on crown
                (22.0, 115.0),
                # Brow tine
                (36.0, 125.0),
                (26.0, 128.0),
                # Main beam upward
                (32.0, 155.0),
                # Second tine
                (45.0, 168.0),
                (33.0, 172.0),
                # Top crown crown fork
                (38.0, 205.0),  # Main tip
                (28.0, 185.0),
                (20.0, 150.0),
                (14.0, 110.0),
                (12.0, 88.0),
            ]
            polygons.append(antlers)

            # Deer Eye aperture
            eyes.append((0.0, 88.0, 4.0))

        elif pose.gesture == PuppetGesture.BIRD:
            # ------------------------------------------------------------------
            # Stylized Bird / Soaring Eagle Silhouette
            # ------------------------------------------------------------------
            # Central Torso, Tail, and Head
            body_pts = [
                # Crown & Sharp Beak
                (0.0, 105.0),   # Beak tip
                (6.0, 92.0),    # Forehead
                (8.0, 75.0),    # Throat
                (18.0, 45.0),   # Right chest
                # Right wing root
                (22.0, 30.0),
                # Lower belly
                (12.0, 0.0),
                # Fanned tail feathers
                (25.0, -45.0),
                (10.0, -38.0),
                (0.0, -50.0),   # Central tail feather
                (-10.0, -38.0),
                (-25.0, -45.0),
                # Left wing root
                (-12.0, 0.0),
                (-22.0, 30.0),
                (-18.0, 45.0),  # Left chest
                (-8.0, 75.0),
                (-6.0, 92.0),
            ]
            polygons.append(body_pts)

            # Dynamic Wings: left and right flight feathers rotate with wing_tilt
            left_rad = math.radians(pose.left_wing_tilt)
            right_rad = math.radians(pose.right_wing_tilt)

            # Left Wing (scalloped primary feathers)
            raw_left_wing = [
                (-22.0, 35.0),
                (-55.0, 50.0),
                (-95.0, 68.0),
                (-140.0, 82.0), # Wingtip primary
                (-125.0, 55.0),
                (-100.0, 35.0),
                (-70.0, 20.0),
                (-35.0, 15.0),
                (-15.0, 20.0),
            ]
            rot_left = []
            for u, v in raw_left_wing:
                # Pivot around (-22.0, 35.0)
                du = u - (-22.0)
                dv = v - 35.0
                ru = -22.0 + (du * math.cos(left_rad) - dv * math.sin(left_rad))
                rv = 35.0 + (du * math.sin(left_rad) + dv * math.cos(left_rad))
                rot_left.append((ru, rv))
            polygons.append(rot_left)

            # Right Wing
            raw_right_wing = [
                (22.0, 35.0),
                (55.0, 50.0),
                (95.0, 68.0),
                (140.0, 82.0),  # Wingtip primary
                (125.0, 55.0),
                (100.0, 35.0),
                (70.0, 20.0),
                (35.0, 15.0),
                (15.0, 20.0),
            ]
            rot_right = []
            for u, v in raw_right_wing:
                # Pivot around (22.0, 35.0)
                du = u - 22.0
                dv = v - 35.0
                ru = 22.0 + (du * math.cos(right_rad) - dv * math.sin(right_rad))
                rv = 35.0 + (du * math.sin(right_rad) + dv * math.cos(right_rad))
                rot_right.append((ru, rv))
            polygons.append(rot_right)

            # Glowing eye apertures on both sides of head
            eyes.append((-3.0, 85.0, 3.5))
            eyes.append((3.0, 85.0, 3.5))

        elif pose.gesture == PuppetGesture.RABBIT:
            # ------------------------------------------------------------------
            # Stylized Rabbit / Bunny Silhouette
            # ------------------------------------------------------------------
            rabbit_pts = [
                # Chest & Front paws
                (-15.0, -15.0),
                (-25.0, 20.0),
                (-20.0, 55.0),
                # Twitchy nose & Muzzle
                (-5.0, 72.0),
                (15.0, 80.0),   # Nose
                (18.0, 70.0),
                (12.0, 60.0),   # Cheek
                # Forehead rising to ears
                (15.0, 90.0),
                # Front tall rabbit ear
                (22.0, 130.0),
                (26.0, 175.0),
                (20.0, 185.0),  # Ear tip
                (10.0, 135.0),
                # Back rabbit ear
                (6.0, 165.0),
                (0.0, 175.0),
                (-4.0, 130.0),
                # Back of neck
                (-15.0, 85.0),
                (-22.0, 45.0),
                (-20.0, -15.0),
            ]
            polygons.append(rabbit_pts)
            # Big curious eye
            eyes.append((4.0, 75.0, 5.0))

        elif pose.gesture == PuppetGesture.SWAN:
            # ------------------------------------------------------------------
            # Stylized Swan / Graceful S-Neck Silhouette
            # ------------------------------------------------------------------
            swan_pts = [
                # Long neck rising from bottom
                (-15.0, -25.0),
                (-25.0, 30.0),
                (-30.0, 80.0),
                (-20.0, 130.0), # Neck curve
                (-5.0, 160.0),
                (12.0, 165.0),  # Crown
                # Tapering bill / beak
                (45.0, 155.0),  # Beak tip
                (20.0, 142.0),
                (8.0, 138.0),   # Throat
                # Inner neck contour
                (-2.0, 115.0),
                (-12.0, 70.0),
                (-8.0, 20.0),
                (5.0, -25.0),
            ]
            polygons.append(swan_pts)
            # Sleek eye aperture
            eyes.append((10.0, 152.0, 4.0))

        return polygons, eyes


# ------------------------------------------------------------------------------
# High-Contrast Head-Up Display (HUD)
# ------------------------------------------------------------------------------
class HUDOverlay:
    """
    Renders high-contrast glassmorphic UI elements:
    - Active gesture status badge with animated confidence meter.
    - FPS counter and frame timing.
    - Interactive hotkey cheat sheet and gesture master guide.
    """

    # UI Theme Palette
    COLOR_BG_CARD = (18, 18, 22)         # Sleek dark slate
    COLOR_BORDER = (55, 55, 65)          # Card outline
    COLOR_ACCENT = (235, 175, 45)        # Electric Amber / Cyan
    COLOR_TEXT_MAIN = (250, 250, 250)    # Crisp white
    COLOR_TEXT_MUTED = (160, 160, 170)   # Slate gray
    COLOR_EMERALD = (85, 200, 110)       # Active indicator

    @classmethod
    def draw_hud(
        cls,
        frame: np.ndarray,
        pose: PuppetPose,
        fps: float,
        shadow_screen_mode: bool,
        show_landmarks: bool,
        show_guide: bool,
        is_live_camera: bool = True,
    ) -> np.ndarray:
        """
        Draws the on-screen UI overlays over the frame.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # ----------------------------------------------------------------------
        # Top-Left: Active Puppet Status Badge
        # ----------------------------------------------------------------------
        cls._draw_status_card(canvas, pose, fps, shadow_screen_mode, show_landmarks, is_live_camera)

        # ----------------------------------------------------------------------
        # Bottom-Right: Interactive Shortcut Keys Bar
        # ----------------------------------------------------------------------
        cls._draw_controls_bar(canvas, w, h, show_guide)

        # ----------------------------------------------------------------------
        # Fullscreen Gesture Cheat Sheet (Toggled with 'h')
        # ----------------------------------------------------------------------
        if show_guide:
            cls._draw_guide_modal(canvas, w, h)

        return canvas

    @classmethod
    def _draw_status_card(
        cls,
        canvas: np.ndarray,
        pose: PuppetPose,
        fps: float,
        shadow_screen: bool,
        landmarks_on: bool,
        is_live_camera: bool = True,
    ) -> None:
        """Draws the top-left status and confidence indicator card."""
        card_w, card_h = 340, 115
        x, y = 20, 20

        # Semi-transparent dark card background
        sub_img = canvas[y : y + card_h, x : x + card_w]
        dark_rect = np.full_like(sub_img, cls.COLOR_BG_CARD)
        cv2.addWeighted(dark_rect, 0.82, sub_img, 0.18, 0, sub_img)

        # Card border
        cv2.rectangle(canvas, (x, y), (x + card_w, y + card_h), cls.COLOR_BORDER, 1, lineType=cv2.LINE_AA)

        # Status Dot (Green if detected, Slate if none)
        is_active = pose.gesture != PuppetGesture.NONE
        dot_color = cls.COLOR_EMERALD if is_active else cls.COLOR_TEXT_MUTED
        cv2.circle(canvas, (x + 18, y + 25), 6, dot_color, -1, lineType=cv2.LINE_AA)

        # Detected Gesture Label
        gesture_title = pose.gesture.value if is_active else "Ready (Form a Hand Puppet)"
        cv2.putText(
            canvas,
            gesture_title,
            (x + 35, y + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            cls.COLOR_TEXT_MAIN,
            1,
            lineType=cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x, bar_y = x + 18, y + 50
        bar_w, bar_h = 240, 8
        conf = pose.confidence if is_active else 0.0

        # Background track
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 48), -1)
        # Active fill
        fill_w = int(bar_w * conf)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), cls.COLOR_ACCENT, -1)

        # Percentage text
        cv2.putText(
            canvas,
            f"{int(conf * 100)}%",
            (bar_x + bar_w + 10, bar_y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            cls.COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )

        # Secondary info row: FPS, Cam status, and Screen mode
        cam_tag = "LIVE CAM" if is_live_camera else "SYNTHETIC DEMO"
        info_str = f"FPS: {fps:.1f}  |  {cam_tag}  |  Mesh: {'ON' if landmarks_on else 'OFF'}"
        cv2.putText(
            canvas,
            info_str,
            (x + 18, y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            cls.COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )

    @classmethod
    def _draw_controls_bar(cls, canvas: np.ndarray, w: int, h: int, guide_open: bool) -> None:
        """Draws a clean bottom shortcut helper bar."""
        bar_text = "[H] Puppet Guide  |  [S] Shadow Theater  |  [L] Landmarks  |  [C] Flip Cam  |  [Q/ESC] Quit"
        cv2.putText(
            canvas,
            bar_text,
            (22, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            cls.COLOR_TEXT_MAIN,
            1,
            lineType=cv2.LINE_AA,
        )

    @classmethod
    def _draw_guide_modal(cls, canvas: np.ndarray, w: int, h: int) -> None:
        """Renders an interactive on-screen Cheat Sheet for all shadow puppet shapes."""
        modal_w, modal_h = min(680, w - 40), min(420, h - 80)
        mx = (w - modal_w) // 2
        my = (h - modal_h) // 2

        # Dimmed backdrop
        sub = canvas[my : my + modal_h, mx : mx + modal_w]
        card = np.full_like(sub, (15, 15, 18))
        cv2.addWeighted(card, 0.92, sub, 0.08, 0, sub)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (80, 80, 95), 2, lineType=cv2.LINE_AA)

        # Modal Header
        cv2.putText(
            canvas,
            "Shadow Puppet Master Guide",
            (mx + 25, my + 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            cls.COLOR_ACCENT,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Press [H] to close this cheat-sheet",
            (mx + modal_w - 240, my + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            cls.COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(canvas, (mx + 25, my + 55), (mx + modal_w - 25, my + 55), (50, 50, 60), 1)

        # Puppet Instructions
        puppets_guide = [
            (
                "1. BIRD / EAGLE",
                "Cross your wrists or lock your thumbs together. Spread all fingers wide like wings.",
                "Wings flap dynamically when you tilt your hands!",
            ),
            (
                "2. DOG / WOLF",
                "Raise your index finger (ear). Extend middle & ring together (snout).",
                "Move your thumb up and down to make the dog bark!",
            ),
            (
                "3. DEER / GOAT",
                "Raise index & pinky high (antler horns). Touch middle & ring fingertips to thumb.",
                "Classic silhouette with majestic procedural antler branching.",
            ),
            (
                "4. RABBIT / BUNNY",
                "Hold index and middle fingers straight up together (ears). Curl ring & pinky down.",
                "Fold thumb over ring finger for cute bunny face.",
            ),
            (
                "5. SWAN / GOOSE",
                "Raise forearm vertically (neck). Pinch thumb and fingertips together into a beak.",
                "Graceful curved S-silhouette with tapering bill.",
            ),
        ]

        row_y = my + 90
        for title, desc, tip in puppets_guide:
            cv2.putText(canvas, title, (mx + 30, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.52, cls.COLOR_TEXT_MAIN, 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, desc, (mx + 30, row_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cls.COLOR_TEXT_MUTED, 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, f"Tip: {tip}", (mx + 30, row_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cls.COLOR_ACCENT, 1, lineType=cv2.LINE_AA)
            row_y += 62


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
        cycle = int(self.t // 5) % 3  # Cycles every 5 seconds

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

        # Build mock 21 landmarks
        px_pts = np.zeros((21, 2), dtype=np.float32)
        px_pts[0] = [cx, cy + scale * 0.5]  # Wrist
        px_pts[9] = [cx, cy - scale * 0.5]  # Middle MCP

        if cycle == 0:
            # Synthetic Deer: Index and Pinky extended, middle/ring touching thumb
            # Index tip
            px_pts[8] = [cx - scale * 0.45, cy - scale * 1.2]
            # Pinky tip
            px_pts[20] = [cx + scale * 0.45, cy - scale * 1.2]
            # Thumb, Middle, Ring pinched together
            px_pts[4] = [cx, cy - scale * 0.7]
            px_pts[12] = [cx, cy - scale * 0.7]
            px_pts[16] = [cx, cy - scale * 0.7]
            fe = {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True}
        elif cycle == 1:
            # Synthetic Dog: Index up, middle & ring extended together, thumb barking
            px_pts[8] = [cx - scale * 0.35, cy - scale * 1.1]
            px_pts[12] = [cx + scale * 0.3, cy - scale * 0.9]
            px_pts[16] = [cx + scale * 0.28, cy - scale * 0.88]
            # Barking thumb
            bark_y = cy - scale * (0.65 - 0.25 * abs(math.sin(self.t * 4.0)))
            px_pts[4] = [cx + scale * 0.25, bark_y]
            px_pts[20] = [cx - scale * 0.2, cy - scale * 0.4]
            fe = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": False}
        else:
            # Synthetic Rabbit: Index & Middle up together
            px_pts[8] = [cx - scale * 0.1, cy - scale * 1.2]
            px_pts[12] = [cx + scale * 0.1, cy - scale * 1.2]
            px_pts[4] = [cx, cy - scale * 0.5]
            px_pts[16] = [cx + scale * 0.1, cy - scale * 0.3]
            px_pts[20] = [cx + scale * 0.2, cy - scale * 0.3]
            fe = {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}

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


# ------------------------------------------------------------------------------
# Unified Hand Tracker (Tasks API + Legacy Solutions Fallback)
# ------------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]


class UnifiedHandTracker:
    """
    Unified hand tracker:
    Supports modern MediaPipe Tasks API (mp.tasks.vision.HandLandmarker) with
    automatic fallback to legacy mp.solutions.hands.
    """

    def __init__(self, model_name: str = "hand_landmarker.task"):
        self.use_tasks_api = False
        self.detector = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(script_dir, model_name)

        if not os.path.exists(model_file):
            logger.info("Downloading %s model asset...", model_name)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_file)
                logger.info("Model asset downloaded to %s", model_file)
            except Exception as e:
                logger.warning("Could not auto-download model: %s", e)

        # 1. Try modern Tasks API first
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            if os.path.exists(model_file):
                base_opts = mp_python.BaseOptions(model_asset_path=model_file)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    num_hands=2,
                    min_hand_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    running_mode=mp_vision.RunningMode.IMAGE,
                )
                self.detector = mp_vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api = True
                logger.info("Initialized modern MediaPipe HandLandmarker Tasks API.")
        except Exception as e:
            logger.warning("MediaPipe Tasks API initialization skipped: %s", e)

        # 2. Try legacy solutions API if Tasks API not available
        if not self.use_tasks_api:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                try:
                    self.detector = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7,
                    )
                    logger.info("Initialized legacy MediaPipe Hands solutions API.")
                except Exception as e:
                    logger.error("Legacy MediaPipe initialization error: %s", e)
            else:
                logger.warning("MediaPipe solutions module not available in this build.")

    def process(self, rgb_frame: np.ndarray) -> List[Tuple[Any, str]]:
        """Extracts hands from an RGB frame returning list of (landmarks_21, handedness_str)."""
        if self.detector is None:
            return []

        results: List[Tuple[Any, str]] = []
        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection = self.detector.detect(mp_image)
            if detection.hand_landmarks:
                for idx, hand_lms in enumerate(detection.hand_landmarks):
                    label = "Right"
                    if detection.handedness and idx < len(detection.handedness):
                        cats = detection.handedness[idx]
                        if cats:
                            label = cats[0].category_name or "Right"
                    results.append((hand_lms, label))
        else:
            res = self.detector.process(rgb_frame)
            if res.multi_hand_landmarks and res.multi_handedness:
                for lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label
                    results.append((lms.landmark, label))

        return results

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks_px: np.ndarray) -> None:
        """Draws clean anti-aliased hand bone connections and joints directly with OpenCV."""
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = (int(landmarks_px[p1_idx][0]), int(landmarks_px[p1_idx][1]))
            pt2 = (int(landmarks_px[p2_idx][0]), int(landmarks_px[p2_idx][1]))
            cv2.line(frame, pt1, pt2, (80, 220, 120), 2, lineType=cv2.LINE_AA)
        for i in range(21):
            pt = (int(landmarks_px[i][0]), int(landmarks_px[i][1]))
            cv2.circle(frame, pt, 4, (45, 120, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    def close(self):
        if self.detector and hasattr(self.detector, "close"):
            self.detector.close()


# ------------------------------------------------------------------------------
# Core Application Controller
# ------------------------------------------------------------------------------
class ShadowPuppetsAI:
    """
    Main Application Engine:
    Initializes camera, manages MediaPipe tracking pipeline, handles keyboard input,
    runs smoothing state machine, and streams interactive 30+ FPS graphics.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        demo_mode: bool = False,
    ):
        self.camera_id = camera_id
        self.target_w = width
        self.target_h = height
        self.demo_mode = demo_mode

        # Interactive State Flags
        self.mirror_camera = True
        self.shadow_screen_mode = False
        self.show_landmarks = False
        self.show_guide = False

        # Unified Hand Tracker Setup
        logger.info("Initializing Hand Tracker pipeline (max_num_hands=2)...")
        self.hand_tracker = UnifiedHandTracker()

        # Pipelines
        self.smoother = TemporalSmoother()
        self.demo_gen = SyntheticDemoGenerator(width=self.target_w, height=self.target_h)

        # Camera handle
        self.cap: Optional[cv2.VideoCapture] = None

    def initialize_camera(self) -> bool:
        """
        Robust camera initialization with index discovery and AVFoundation support on macOS.
        """
        if self.demo_mode:
            logger.info("Running in Demo Mode (--demo). Camera capture bypassed.")
            return True

        candidate_indices = [self.camera_id, 0, 1, 2]
        unique_indices = list(dict.fromkeys(candidate_indices))

        # On macOS, prioritize AVFoundation backend explicitly
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]

        for backend in backends:
            backend_name = "AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "DEFAULT"
            for idx in unique_indices:
                try:
                    logger.info("Attempting camera index %d with %s backend...", idx, backend_name)
                    cap = cv2.VideoCapture(idx, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        # Verify frame acquisition directly before modifying capture settings
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cap = cap
                            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            logger.info("✓ Camera index %d successfully acquired at %dx%d (%s)!", idx, actual_w, actual_h, backend_name)
                            return True
                        cap.release()
                except Exception as e:
                    logger.warning("Error probing camera index %d (%s): %s", idx, backend_name, str(e))

        logger.error("No functional webcam device could be opened.")
        return False

    def run(self) -> None:
        """
        Main application event loop.
        """
        cam_ok = self.initialize_camera()
        if not cam_ok:
            logger.warning("Falling back to Synthetic Demonstration Mode for interactive testing.")
            self.demo_mode = True

        window_name = "Hand-Gesture Shadow Puppets AI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.target_w, self.target_h)

        fps_tracker = 30.0
        prev_time = time.perf_counter()

        logger.info("Starting Shadow Puppets AI stream. Press [Q] or [ESC] to quit, [H] for guide.")

        try:
            while True:
                curr_time = time.perf_counter()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_tracker = 0.9 * fps_tracker + 0.1 * (1.0 / dt)

                # --------------------------------------------------------------
                # Frame Acquisition
                # --------------------------------------------------------------
                if self.demo_mode:
                    frame, extracted_hands = self.demo_gen.generate_frame_and_landmarks()
                else:
                    assert self.cap is not None
                    success, frame = self.cap.read()
                    if not success or frame is None:
                        logger.warning("Empty camera frame received. Re-probing...")
                        time.sleep(0.01)
                        continue

                    if self.mirror_camera:
                        frame = cv2.flip(frame, 1)

                    h, w = frame.shape[:2]

                    # ----------------------------------------------------------
                    # MediaPipe Landmark Processing
                    # ----------------------------------------------------------
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected_hands = self.hand_tracker.process(rgb_frame)

                    extracted_hands: List[HandLandmarksData] = []
                    for lms, label in detected_hands:
                        # Adjust label if camera is mirrored
                        if self.mirror_camera:
                            label = "Left" if label == "Right" else "Right"

                        hand_data = HandFeatureExtractor.process_hand(
                            landmarks=lms,
                            handedness_label=label,
                            frame_width=w,
                            frame_height=h,
                        )
                        extracted_hands.append(hand_data)

                        # Optional Skeletal Landmark Mesh Rendering
                        if self.show_landmarks:
                            UnifiedHandTracker.draw_skeleton(frame, hand_data.pixel_landmarks)

                # --------------------------------------------------------------
                # Gesture Evaluation & Temporal Smoothing
                # --------------------------------------------------------------
                raw_pose = GestureRecognizer.evaluate(extracted_hands)
                smoothed_pose, active_alpha = self.smoother.update(raw_pose)

                # --------------------------------------------------------------
                # Procedural Shadow Puppet Silhouette Rendering
                # --------------------------------------------------------------
                frame = PuppetRenderer.render(
                    frame=frame,
                    pose=smoothed_pose,
                    alpha=active_alpha,
                    shadow_screen_mode=self.shadow_screen_mode,
                )

                # --------------------------------------------------------------
                # High-Contrast Head-Up Display (HUD)
                # --------------------------------------------------------------
                frame = HUDOverlay.draw_hud(
                    frame=frame,
                    pose=smoothed_pose,
                    fps=fps_tracker,
                    shadow_screen_mode=self.shadow_screen_mode,
                    show_landmarks=self.show_landmarks,
                    show_guide=self.show_guide,
                    is_live_camera=(not self.demo_mode),
                )

                # Display frame
                cv2.imshow(window_name, frame)

                # --------------------------------------------------------------
                # Interactive Keyboard Controls
                # --------------------------------------------------------------
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):  # ESC or Q
                    logger.info("Exit key pressed. Shutting down...")
                    break
                elif key in (ord("s"), ord("S")):
                    self.shadow_screen_mode = not self.shadow_screen_mode
                    logger.info("Shadow Screen Mode toggled: %s", self.shadow_screen_mode)
                elif key in (ord("l"), ord("L")):
                    self.show_landmarks = not self.show_landmarks
                    logger.info("Landmarks Overlay toggled: %s", self.show_landmarks)
                elif key in (ord("h"), ord("H")):
                    self.show_guide = not self.show_guide
                    logger.info("Gesture Cheat Sheet toggled: %s", self.show_guide)
                elif key in (ord("c"), ord("C")):
                    self.mirror_camera = not self.mirror_camera
                    logger.info("Mirror Camera toggled: %s", self.mirror_camera)

        except KeyboardInterrupt:
            logger.info("Application interrupted by user. Cleaning up...")
        except Exception as e:
            logger.exception("Unexpected error in main processing loop: %s", str(e))
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """
        Releases camera handles and closes OpenCV display windows.
        """
        logger.info("Releasing hardware and graphics resources...")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.hand_tracker.close()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed successfully.")


# ------------------------------------------------------------------------------
# Entry Point CLI
# ------------------------------------------------------------------------------
def main() -> None:
    """CLI argument parsing and main entry point."""
    parser = argparse.ArgumentParser(
        description="Hand-Gesture Shadow Puppets AI: Real-Time CV & Procedural Shadow Puppetry"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture frame width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture frame height (default: 720)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic animated demo mode (useful for testing without physical camera)",
    )

    args = parser.parse_args()

    app = ShadowPuppetsAI(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        demo_mode=args.demo,
    )
    app.run()


if __name__ == "__main__":
    main()
