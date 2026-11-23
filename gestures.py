"""Gesture data models, hand-feature extraction, recognition and temporal smoothing."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


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
    # Morphing / Cross-fade state between gestures
    prev_gesture: PuppetGesture = PuppetGesture.NONE
    morph_t: float = 1.0  # 1.0 = active gesture, 0.0 = previous gesture
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

        # Hands should point towards opposing lateral directions (or crossed with connected thumbs)
        h1_dir_x = hand1.axis_vector[0]
        h2_dir_x = hand2.axis_vector[0]
        opposing_spread = (h1_dir_x * h2_dir_x < 0.0) or (abs(h1_dir_x - h2_dir_x) > 0.5)

        if base_connected and (h1_ext_count >= 3) and (h2_ext_count >= 3) and (opposing_spread or thumb_dist < 0.7):
            # Confidence score calculation
            conf = 0.72
            if wrist_dist < 0.9 or thumb_dist < 0.6:
                conf += 0.12
            if opposing_spread:
                conf += 0.08
            if h1_ext_count == 4 and h2_ext_count == 4:
                conf += 0.06
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

        # Require thumb to form the tapered muzzle with middle or ring finger
        thumb_muzzle_contact = touch_thumb_middle or touch_thumb_ring
        if thumb_muzzle_contact and antler_spread:
            conf = 0.75
            if touch_thumb_middle and touch_thumb_ring:
                conf += 0.12
            if nd["index_pinky"] > 0.85:
                conf += 0.08
            if touch_middle_ring:
                conf += 0.03
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
        # Thumb should be folded over curled fingers, not pinched to the ear tips (which forms a Swan beak)
        thumb_not_pinched = nd["thumb_index"] > 0.35 or nd["thumb_middle"] > 0.35

        if ears_up and paws_curled and ears_parallel and thumb_not_pinched:
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
        fe = hand.finger_extended
        nd = hand.norm_distances

        # Pinch between Thumb, Index, and Middle forming the tapered beak
        beak_pinch = (nd["thumb_index"] < 0.35) and (nd["thumb_middle"] < 0.38)
        snout_tight = nd["index_middle"] < 0.30
        # Forearm / neck should be held upright, with pinky not extended straight
        hand_upright = abs(hand.rotation_deg) < 65.0
        pinky_relaxed = not fe["pinky"]

        if beak_pinch and snout_tight and hand_upright and pinky_relaxed:
            conf = 0.72
            if nd["thumb_index"] < 0.22:
                conf += 0.12
            if abs(hand.rotation_deg) < 30.0:
                conf += 0.08
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
        - Palm tilted laterally (not flat upright high-five).
        """
        fe = hand.finger_extended
        nd = hand.norm_distances

        wings_open = fe["index"] and fe["middle"] and fe["ring"] and fe["pinky"]
        thumb_head = fe["thumb"] and (nd["thumb_index"] > 0.60)
        lateral_tilt = abs(hand.rotation_deg) > 20.0

        if wings_open and thumb_head and lateral_tilt:
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
        self.prev_gesture = PuppetGesture.NONE
        self.morph_t = 1.0

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
            if self.candidate_gesture != self.current_gesture:
                if self.current_gesture != PuppetGesture.NONE and self.candidate_gesture != PuppetGesture.NONE:
                    self.prev_gesture = self.current_gesture
                    self.morph_t = 0.0
                self.current_gesture = self.candidate_gesture

        # Progress morph_t smoothly between gestures
        if self.morph_t < 1.0:
            self.morph_t = min(1.0, self.morph_t + (1.0 / self.fade_frames))
            if self.morph_t >= 1.0:
                self.prev_gesture = PuppetGesture.NONE

        # 2. Alpha fade-in / fade-out
        if self.current_gesture != PuppetGesture.NONE:
            self.active_alpha = min(1.0, self.active_alpha + (1.0 / self.fade_frames))
        else:
            self.active_alpha = max(0.0, self.active_alpha - (1.0 / self.fade_frames))

        # 3. Low-Pass Kinematic Smoothing
        # Only update kinematics when a valid pose is detected, preserving the last
        # known anchor and scale during fade-out to prevent shrinking or drifting to (0, 0).
        if raw_pose.gesture != PuppetGesture.NONE:
            raw_pos = np.array(raw_pose.center_px, dtype=np.float32)
            if self.active_alpha <= 0.01 or np.linalg.norm(self.smoothed_pos) < 1.0:
                self.smoothed_pos = raw_pos
                self.smoothed_scale = raw_pose.scale_factor
                rad = math.radians(raw_pose.rotation_deg)
                self.smoothed_cos = math.cos(rad)
                self.smoothed_sin = math.sin(rad)
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

            # Dynamic Kinematic Parameters
            self.smoothed_jaw = 0.4 * raw_pose.jaw_open + 0.6 * self.smoothed_jaw
            self.smoothed_left_wing = 0.4 * raw_pose.left_wing_tilt + 0.6 * self.smoothed_left_wing
            self.smoothed_right_wing = 0.4 * raw_pose.right_wing_tilt + 0.6 * self.smoothed_right_wing
            self.flip_horizontal = raw_pose.flip_horizontal
            self.confidence = raw_pose.confidence
        else:
            self.confidence = max(0.0, self.confidence - (1.0 / self.fade_frames))

        smoothed_angle_deg = math.degrees(math.atan2(self.smoothed_sin, self.smoothed_cos))

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
            prev_gesture=self.prev_gesture,
            morph_t=float(self.morph_t),
        )

        return smoothed_pose, self.active_alpha
