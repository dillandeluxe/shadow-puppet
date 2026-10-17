#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Hand-Gesture Shadow Puppets AI.
Validates:
1. Geometric heuristic calculations for each puppet shape (Bird, Dog, Deer, Rabbit, Swan).
2. Scale-invariance (verifying that moving closer/further from the camera preserves recognition).
3. Temporal smoothing, EMA filtering, and cross-fade state transitions.
4. Procedural 2D silhouette rendering and affine polygon math.
5. End-to-end headless pipeline execution with synthetic animation.
"""

import math
import os
import sys
import unittest

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from shadow_puppets import (
    HandFeatureExtractor,
    HandLandmarksData,
    GestureRecognizer,
    PuppetGesture,
    PuppetPose,
    PuppetRenderer,
    TemporalSmoother,
    SyntheticDemoGenerator,
)


class TestShadowPuppets(unittest.TestCase):
    """Test suite covering core computer vision heuristics and graphics routines."""

    def _create_mock_hand(
        self,
        wrist=(640.0, 500.0),
        hand_scale=120.0,
        rotation_deg=0.0,
        extended_fingers=None,
        touch_pairs=None,
        label="Right",
    ) -> HandLandmarksData:
        """Helper to synthesize a HandLandmarksData object with precise geometry."""
        if extended_fingers is None:
            extended_fingers = {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

        wx, wy = wrist
        rad = math.radians(rotation_deg)
        fwd_x = math.sin(rad)
        fwd_y = -math.cos(rad)
        right_x = math.cos(rad)
        right_y = math.sin(rad)

        px_pts = np.zeros((21, 2), dtype=np.float32)
        px_pts[HandFeatureExtractor.WRIST] = [wx, wy]

        # Middle MCP anchor
        mid_mcp_x = wx + hand_scale * fwd_x
        mid_mcp_y = wy + hand_scale * fwd_y
        px_pts[HandFeatureExtractor.MIDDLE_MCP] = [mid_mcp_x, mid_mcp_y]

        # Base MCP layout (strictly scale-proportional)
        px_pts[HandFeatureExtractor.INDEX_MCP] = [mid_mcp_x - hand_scale * 0.25 * right_x, mid_mcp_y]
        px_pts[HandFeatureExtractor.RING_MCP] = [mid_mcp_x + hand_scale * 0.22 * right_x, mid_mcp_y]
        px_pts[HandFeatureExtractor.PINKY_MCP] = [mid_mcp_x + hand_scale * 0.45 * right_x, mid_mcp_y]
        px_pts[HandFeatureExtractor.THUMB_MCP] = [wx - hand_scale * 0.25 * right_x + hand_scale * 0.25 * fwd_x, wy - hand_scale * 0.25 * right_y + hand_scale * 0.25 * fwd_y]

        # Natural lateral spread: splay outward if pinky extended (deer horns), close if rabbit/dog
        idx_spread = -0.35 if extended_fingers["pinky"] else 0.05
        pinky_spread = 0.35 if extended_fingers["pinky"] else 0.0

        finger_specs = [
            ("index", HandFeatureExtractor.INDEX_MCP, HandFeatureExtractor.INDEX_TIP, idx_spread),
            ("middle", HandFeatureExtractor.MIDDLE_MCP, HandFeatureExtractor.MIDDLE_TIP, 0.0),
            ("ring", HandFeatureExtractor.RING_MCP, HandFeatureExtractor.RING_TIP, 0.0),
            ("pinky", HandFeatureExtractor.PINKY_MCP, HandFeatureExtractor.PINKY_TIP, pinky_spread),
        ]

        for name, mcp_idx, tip_idx, lat_spread in finger_specs:
            base_mcp = px_pts[mcp_idx]
            dist_tip = hand_scale * (1.15 if extended_fingers[name] else 0.35)
            px_pts[tip_idx] = [
                base_mcp[0] + dist_tip * fwd_x + hand_scale * lat_spread * right_x,
                base_mcp[1] + dist_tip * fwd_y + hand_scale * lat_spread * right_y,
            ]

        # Thumb tip
        if extended_fingers["thumb"]:
            px_pts[HandFeatureExtractor.THUMB_TIP] = [
                wx - hand_scale * 0.45 * right_x + hand_scale * 0.7 * fwd_x,
                wy - hand_scale * 0.45 * right_y + hand_scale * 0.7 * fwd_y,
            ]
        else:
            px_pts[HandFeatureExtractor.THUMB_TIP] = [
                mid_mcp_x - hand_scale * 0.15 * right_x + hand_scale * 0.1 * fwd_x,
                mid_mcp_y - hand_scale * 0.15 * right_y + hand_scale * 0.1 * fwd_y,
            ]

        # Override touch pairs if specified
        if touch_pairs:
            for src_idx, dst_idx in touch_pairs:
                px_pts[src_idx] = px_pts[dst_idx].copy()

        # Fill PIP and DIP linearly
        for mcp_idx, pip_idx, dip_idx, tip_idx in [
            (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)
        ]:
            px_pts[pip_idx] = px_pts[mcp_idx] + (px_pts[tip_idx] - px_pts[mcp_idx]) * 0.4
            px_pts[dip_idx] = px_pts[mcp_idx] + (px_pts[tip_idx] - px_pts[mcp_idx]) * 0.75

        # Compute normalized distances
        def n_dist(i1, i2):
            return float(np.linalg.norm(px_pts[i1] - px_pts[i2]) / hand_scale)

        norm_dists = {
            "thumb_index": n_dist(4, 8),
            "thumb_middle": n_dist(4, 12),
            "thumb_ring": n_dist(4, 16),
            "thumb_pinky": n_dist(4, 20),
            "index_middle": n_dist(8, 12),
            "middle_ring": n_dist(12, 16),
            "ring_pinky": n_dist(16, 20),
            "index_pinky": n_dist(8, 20),
            "index_tip_mcp": n_dist(8, 5),
            "middle_tip_mcp": n_dist(12, 9),
            "ring_tip_mcp": n_dist(16, 13),
            "pinky_tip_mcp": n_dist(20, 17),
        }

        palm_center = np.mean(
            [px_pts[0], px_pts[5], px_pts[9], px_pts[17]], axis=0
        ).astype(np.float32)

        return HandLandmarksData(
            label=label,
            raw_landmarks=[(p[0] / 1280.0, p[1] / 720.0, 0.0) for p in px_pts],
            pixel_landmarks=px_pts,
            wrist_px=px_pts[0],
            middle_mcp_px=px_pts[9],
            palm_center_px=palm_center,
            hand_length_px=hand_scale,
            axis_vector=np.array([fwd_x, fwd_y], dtype=np.float32),
            rotation_deg=rotation_deg,
            finger_extended=extended_fingers,
            norm_distances=norm_dists,
        )

    def test_deer_heuristic(self):
        """Test deer recognition: Index and Pinky extended, middle and ring touching thumb."""
        deer_hand = self._create_mock_hand(
            extended_fingers={"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True},
            touch_pairs=[(12, 4), (16, 4)],  # Middle and Ring touching Thumb
        )
        pose = GestureRecognizer.evaluate([deer_hand])
        self.assertEqual(pose.gesture, PuppetGesture.DEER)
        self.assertGreaterEqual(pose.confidence, 0.70)

    def test_dog_heuristic_and_jaw_kinematics(self):
        """Test dog recognition and dynamic openable jaw factor."""
        # Closed jaw: thumb tip positioned right at middle/ring tips (lower jaw touching snout)
        dog_hand_closed = self._create_mock_hand(
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": False},
            touch_pairs=[(16, 12)],  # Middle and Ring extended together
        )
        dog_hand_closed.pixel_landmarks[4] = dog_hand_closed.pixel_landmarks[12] - dog_hand_closed.hand_length_px * 0.15 * dog_hand_closed.axis_vector
        dog_hand_closed.norm_distances["thumb_middle"] = float(
            np.linalg.norm(dog_hand_closed.pixel_landmarks[4] - dog_hand_closed.pixel_landmarks[12])
            / dog_hand_closed.hand_length_px
        )
        pose_closed = GestureRecognizer.evaluate([dog_hand_closed])
        self.assertEqual(pose_closed.gesture, PuppetGesture.DOG)
        self.assertGreaterEqual(pose_closed.confidence, 0.70)
        self.assertAlmostEqual(pose_closed.jaw_open, 0.0, delta=0.05)

        # Open jaw: Thumb drops down toward palm / wrist (mouth opens)
        dog_hand_open = self._create_mock_hand(
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": False},
            touch_pairs=[(16, 12)],
        )
        dog_hand_open.pixel_landmarks[4] = dog_hand_open.pixel_landmarks[12] - dog_hand_open.hand_length_px * 0.60 * dog_hand_open.axis_vector
        dog_hand_open.norm_distances["thumb_middle"] = float(
            np.linalg.norm(dog_hand_open.pixel_landmarks[4] - dog_hand_open.pixel_landmarks[12])
            / dog_hand_open.hand_length_px
        )
        pose_open = GestureRecognizer.evaluate([dog_hand_open])
        self.assertEqual(pose_open.gesture, PuppetGesture.DOG)
        self.assertGreater(pose_open.jaw_open, pose_closed.jaw_open)

    def test_rabbit_heuristic(self):
        """Test rabbit recognition: Index & Middle up parallel, Ring & Pinky down."""
        rabbit_hand = self._create_mock_hand(
            extended_fingers={"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False},
        )
        pose = GestureRecognizer.evaluate([rabbit_hand])
        self.assertEqual(pose.gesture, PuppetGesture.RABBIT)
        self.assertGreaterEqual(pose.confidence, 0.70)

    def test_two_hand_bird_heuristic(self):
        """Test classic two-handed bird with crossing wrists/thumbs and spread wings."""
        hand1 = self._create_mock_hand(
            wrist=(600.0, 400.0),
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
            rotation_deg=-30.0,
            label="Left",
        )
        hand2 = self._create_mock_hand(
            wrist=(620.0, 400.0),
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
            rotation_deg=30.0,
            label="Right",
        )
        pose = GestureRecognizer.evaluate([hand1, hand2])
        self.assertEqual(pose.gesture, PuppetGesture.BIRD)
        self.assertGreaterEqual(pose.confidence, 0.70)

    def test_scale_invariance(self):
        """Verifies that recognition is invariant to distance from camera (hand scale)."""
        scales = [60.0, 100.0, 180.0, 250.0]
        for s in scales:
            hand = self._create_mock_hand(
                hand_scale=s,
                extended_fingers={"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True},
                touch_pairs=[(12, 4), (16, 4)],
            )
            pose = GestureRecognizer.evaluate([hand])
            self.assertEqual(pose.gesture, PuppetGesture.DEER, f"Failed at scale {s}")

    def test_temporal_smoother(self):
        """Validates EMA filtering, state debouncing, and cross-fade blending."""
        smoother = TemporalSmoother(debounce_frames=3, fade_frames=10)

        raw_deer = PuppetPose(
            gesture=PuppetGesture.DEER,
            confidence=0.85,
            center_px=(640.0, 360.0),
            scale_factor=100.0,
            rotation_deg=10.0,
        )

        # Frame 1: Candidate registered, not yet active
        pose1, alpha1 = smoother.update(raw_deer)
        self.assertEqual(pose1.gesture, PuppetGesture.NONE)

        # Frame 2: Still candidate
        pose2, alpha2 = smoother.update(raw_deer)
        self.assertEqual(pose2.gesture, PuppetGesture.NONE)

        # Frame 3: Debounce threshold reached -> state activates
        pose3, alpha3 = smoother.update(raw_deer)
        self.assertEqual(pose3.gesture, PuppetGesture.DEER)
        self.assertGreater(alpha3, 0.0)

        # Feed 15 frames: Alpha should ramp up to 1.0
        for _ in range(15):
            pose_n, alpha_n = smoother.update(raw_deer)
        self.assertAlmostEqual(alpha_n, 1.0, places=2)

    def test_procedural_rendering(self):
        """Ensures silhouette rasterization executes cleanly without exceptions."""
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for g in [PuppetGesture.DOG, PuppetGesture.DEER, PuppetGesture.BIRD, PuppetGesture.RABBIT, PuppetGesture.SWAN]:
            pose = PuppetPose(
                gesture=g,
                confidence=0.9,
                center_px=(640.0, 360.0),
                scale_factor=120.0,
                rotation_deg=15.0,
                jaw_open=0.5,
                left_wing_tilt=10.0,
                right_wing_tilt=-10.0,
            )
            rendered = PuppetRenderer.render(test_frame.copy(), pose, alpha=1.0, shadow_screen_mode=False)
            self.assertEqual(rendered.shape, test_frame.shape)
            # Verify pixels were written
            self.assertGreater(np.sum(rendered), 0, f"No pixels rendered for {g}")

    def test_synthetic_demo_pipeline(self):
        """Tests that SyntheticDemoGenerator drives the pipeline flawlessly."""
        gen = SyntheticDemoGenerator(width=640, height=360)
        smoother = TemporalSmoother(debounce_frames=2)

        for _ in range(30):
            frame, hands = gen.generate_frame_and_landmarks()
            raw_pose = GestureRecognizer.evaluate(hands)
            smoothed_pose, alpha = smoother.update(raw_pose)
            out = PuppetRenderer.render(frame, smoothed_pose, alpha)
            self.assertEqual(out.shape, (360, 640, 3))


if __name__ == "__main__":
    unittest.main()
