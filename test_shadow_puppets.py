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
import tempfile
import unittest

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from demo import SyntheticDemoGenerator
from gestures import (
    GestureRecognizer,
    HandFeatureExtractor,
    HandLandmarksData,
    PuppetGesture,
    PuppetPose,
    TemporalSmoother,
)
from rendering import HUDOverlay, PuppetRenderer
from shadow_puppets import ShadowPuppetsAI


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

    def test_swan_heuristic(self):
        """Test swan recognition: Thumb, Index, Middle pinched together into a beak."""
        swan_hand = self._create_mock_hand(
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": False, "pinky": False},
            touch_pairs=[(8, 4), (12, 4)],
            rotation_deg=0.0,
        )
        pose = GestureRecognizer.evaluate([swan_hand])
        self.assertEqual(pose.gesture, PuppetGesture.SWAN)
        self.assertGreaterEqual(pose.confidence, 0.70)

    def test_single_hand_bird_heuristic(self):
        """Test single-hand dove: Thumb up, 4 fingers spread wide, palm tilted."""
        bird_hand = self._create_mock_hand(
            extended_fingers={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
            rotation_deg=35.0,
        )
        pose = GestureRecognizer.evaluate([bird_hand])
        self.assertEqual(pose.gesture, PuppetGesture.BIRD)
        self.assertGreaterEqual(pose.confidence, 0.70)

    def test_rock_on_does_not_trigger_deer(self):
        """Negative test: Rock-on / Horns sign (thumb folded away, not touching muzzle) must NOT trigger Deer."""
        rock_hand = self._create_mock_hand(
            extended_fingers={"thumb": False, "index": True, "middle": False, "ring": False, "pinky": True},
            touch_pairs=[(12, 16)],
        )
        rock_hand.norm_distances["thumb_middle"] = 0.85
        rock_hand.norm_distances["thumb_ring"] = 0.80
        pose = GestureRecognizer.evaluate([rock_hand])
        self.assertNotEqual(pose.gesture, PuppetGesture.DEER)

    def test_hand_loss_preserves_position(self):
        """Verifies that when hands exit the frame, position stays anchored instead of flying to (0,0)."""
        smoother = TemporalSmoother(debounce_frames=1, fade_frames=10)
        valid_pose = PuppetPose(
            gesture=PuppetGesture.DOG,
            confidence=0.9,
            center_px=(640.0, 360.0),
            scale_factor=150.0,
        )
        smoother.update(valid_pose)

        none_pose = PuppetPose(gesture=PuppetGesture.NONE, confidence=0.0)
        smoothed, _ = smoother.update(none_pose)

        self.assertAlmostEqual(smoothed.center_px[0], 640.0, delta=1.0)
        self.assertAlmostEqual(smoothed.center_px[1], 360.0, delta=1.0)
        self.assertAlmostEqual(smoothed.scale_factor, 150.0, delta=1.0)

    def test_symmetric_wing_flapping(self):
        """Verifies that positive wing tilts raise both left and right wingtips upward."""
        pose_flat = PuppetPose(
            gesture=PuppetGesture.BIRD,
            confidence=0.9,
            left_wing_tilt=0.0,
            right_wing_tilt=0.0,
        )
        pose_up = PuppetPose(
            gesture=PuppetGesture.BIRD,
            confidence=0.9,
            left_wing_tilt=20.0,
            right_wing_tilt=20.0,
        )
        poly_flat, _ = PuppetRenderer._get_canonical_geometry(pose_flat)
        poly_up, _ = PuppetRenderer._get_canonical_geometry(pose_up)

        # poly[1] is left wing, poly[2] is right wing
        left_wingtip_flat = max(p[1] for p in poly_flat[1])
        left_wingtip_up = max(p[1] for p in poly_up[1])
        right_wingtip_flat = max(p[1] for p in poly_flat[2])
        right_wingtip_up = max(p[1] for p in poly_up[2])

        self.assertGreater(left_wingtip_up, left_wingtip_flat, "Left wing did not lift upward with positive tilt")
        self.assertGreater(right_wingtip_up, right_wingtip_flat, "Right wing did not lift upward with positive tilt")

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

    def test_gesture_transition_crossfade(self):
        """Validates that transitioning directly between two gestures triggers smooth morphing."""
        smoother = TemporalSmoother(debounce_frames=2, fade_frames=10)
        raw_deer = PuppetPose(gesture=PuppetGesture.DEER, confidence=0.9, center_px=(640.0, 360.0), scale_factor=100.0)
        for _ in range(5):
            smoother.update(raw_deer)

        # Transition to Dog
        raw_dog = PuppetPose(gesture=PuppetGesture.DOG, confidence=0.9, center_px=(640.0, 360.0), scale_factor=100.0)
        smoother.update(raw_dog)  # Frame 1 debounce
        pose_trans, _ = smoother.update(raw_dog)  # Frame 2 debounce triggers transition

        self.assertEqual(pose_trans.gesture, PuppetGesture.DOG)
        self.assertEqual(pose_trans.prev_gesture, PuppetGesture.DEER)
        self.assertLess(pose_trans.morph_t, 1.0)

        # Test rendering during morph transition
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = PuppetRenderer.render(frame, pose_trans, alpha=1.0)
        self.assertEqual(rendered.shape, (720, 1280, 3))

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

    def test_camera_discovery(self):
        """Validates that camera discovery probes return properly structured records."""
        cameras = ShadowPuppetsAI.discover_available_cameras(max_probe=2)
        self.assertIsInstance(cameras, list)
        for cam in cameras:
            self.assertIn("index", cam)
            self.assertIn("width", cam)
            self.assertIn("height", cam)
            self.assertIn("fps", cam)
            self.assertIn("backend", cam)
            self.assertIsInstance(cam["index"], int)
            self.assertGreater(cam["width"], 0)
            self.assertGreater(cam["height"], 0)

    def test_toast_system(self):
        """Validates HUD toast message scheduling and expiration."""
        app = ShadowPuppetsAI(demo_mode=True)
        self.assertIsNone(app.toast_message)
        app.set_toast("Testing Toast Notification", duration=1.0)
        self.assertEqual(app.toast_message, "Testing Toast Notification")
        self.assertGreater(app.toast_expiry, 0.0)
        app.cleanup()

    def test_camera_switching_state(self):
        """Validates camera switching state logic and demo mode handling."""
        app = ShadowPuppetsAI(camera_id=0, demo_mode=True)
        # Mock available cameras
        app.available_cameras = [0, 1]
        # In demo mode, switching with mock/real camera
        app.switch_camera(target_cam_id=1)
        self.assertIn(app.camera_id, [0, 1])
        app.cleanup()

    def test_hud_with_toast_and_camera_info(self):
        """Verifies HUD rendering cleanly incorporates camera info and floating banner toasts."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pose = PuppetPose(gesture=PuppetGesture.BIRD, confidence=0.88, center_px=(640.0, 360.0), scale_factor=120.0)
        hud_frame = HUDOverlay.draw_hud(
            frame=frame,
            pose=pose,
            fps=30.0,
            shadow_screen_mode=False,
            show_landmarks=False,
            show_guide=False,
            is_live_camera=True,
            camera_info="CAM #1",
            toast_message="Switched to Camera #1",
        )
        self.assertEqual(hud_frame.shape, (720, 1280, 3))
        # Ensure HUD drew content
        self.assertGreater(np.sum(hud_frame), 0)


class TestRobustness(unittest.TestCase):
    """Regression tests for audit findings: imports, download, tracker and camera failures."""

    def test_type_hints_resolve(self):
        """Python < 3.14 evaluates hints at import; a missing `Any` import crashed there."""
        import typing
        import shadow_puppets as sp
        typing.get_type_hints(sp.HandFeatureExtractor.process_hand)
        typing.get_type_hints(sp.UnifiedHandTracker.process)

    def test_failed_download_leaves_no_partial_model(self):
        import tempfile
        from unittest import mock
        import shadow_puppets as sp

        def broken_download(url, dest):
            with open(dest, "wb") as f:
                f.write(b"partial")
            raise OSError("network dropped")

        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.task")
            with mock.patch("urllib.request.urlretrieve", broken_download):
                sp.UnifiedHandTracker(model).close()
            self.assertEqual(os.listdir(d), [])

    def test_no_tracker_falls_back_to_demo(self):
        from unittest import mock
        import shadow_puppets as sp

        stub = mock.Mock(detector=None)
        with mock.patch.object(sp, "UnifiedHandTracker", return_value=stub):
            self.assertTrue(sp.ShadowPuppetsAI(demo_mode=False).demo_mode)

    def test_dead_camera_falls_back_to_demo(self):
        from unittest import mock
        import shadow_puppets as sp

        class DeadCap:
            reads = 0
            def read(self):
                DeadCap.reads += 1
                if DeadCap.reads > 500:
                    raise RuntimeError("loop never gave up")
                return False, None
            def release(self):
                pass

        with mock.patch.object(sp, "UnifiedHandTracker", return_value=mock.Mock(detector=object())):
            app = sp.ShadowPuppetsAI(width=320, height=180)
        app.cap = DeadCap()
        with mock.patch.object(app, "initialize_camera", return_value=True), \
                mock.patch.object(sp.cv2, "namedWindow"), mock.patch.object(sp.cv2, "resizeWindow"), \
                mock.patch.object(sp.cv2, "imshow"), mock.patch.object(sp.cv2, "destroyAllWindows"), \
                mock.patch.object(sp.cv2, "waitKey", return_value=ord("q")):
            app.run()
        self.assertTrue(app.demo_mode)

    def test_camera_gets_requested_size(self):
        from unittest import mock
        import shadow_puppets as sp

        sets = []

        class FakeCap:
            def __init__(self, *a): pass
            def isOpened(self): return True
            def read(self): return True, np.zeros((4, 4, 3), np.uint8)
            def get(self, prop): return 0
            def set(self, prop, val): sets.append((prop, val))
            def release(self): pass

        with mock.patch.object(sp, "UnifiedHandTracker", return_value=mock.Mock(detector=object())):
            app = sp.ShadowPuppetsAI(width=640, height=360)
        with mock.patch.object(sp.cv2, "VideoCapture", FakeCap):
            self.assertTrue(app.initialize_camera())
        self.assertIn((sp.cv2.CAP_PROP_FRAME_WIDTH, 640), sets)
        self.assertIn((sp.cv2.CAP_PROP_FRAME_HEIGHT, 360), sets)


if __name__ == "__main__":
    unittest.main()
