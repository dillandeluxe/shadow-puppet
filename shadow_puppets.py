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
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
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


from demo import SyntheticDemoGenerator
from gestures import (
    GestureRecognizer,
    HandFeatureExtractor,
    HandLandmarksData,
    TemporalSmoother,
)
from rendering import HUDOverlay, PuppetRenderer


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
        self._last_timestamp_ms = 0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(script_dir, model_name)

        if not os.path.exists(model_file):
            logger.info("Downloading %s model asset...", model_name)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            tmp_model_file = model_file + ".tmp"
            try:
                urllib.request.urlretrieve(url, tmp_model_file)
                if os.path.exists(tmp_model_file) and os.path.getsize(tmp_model_file) > 1000000:
                    os.replace(tmp_model_file, model_file)
                    logger.info("Model asset downloaded to %s", model_file)
                elif os.path.exists(tmp_model_file):
                    os.remove(tmp_model_file)
            except Exception as e:
                logger.warning("Could not auto-download model: %s", e)
                if os.path.exists(tmp_model_file):
                    try:
                        os.remove(tmp_model_file)
                    except OSError:
                        pass

        # 1. Try modern Tasks API first with real-time video tracking
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
                    running_mode=mp_vision.RunningMode.VIDEO,
                )
                self.detector = mp_vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api = True
                logger.info("Initialized modern MediaPipe HandLandmarker Tasks API (Video Tracking Mode).")
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
            if hasattr(self.detector, "detect_for_video"):
                now_ms = max(self._last_timestamp_ms + 1, int(time.perf_counter() * 1000))
                self._last_timestamp_ms = now_ms
                detection = self.detector.detect_for_video(mp_image, now_ms)
            else:
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

        # Multi-camera discovery and Toast state
        self.available_cameras: List[int] = []
        self.toast_message: Optional[str] = None
        self.toast_expiry: float = 0.0

        # Interactive State Flags
        self.mirror_camera = True
        self.shadow_screen_mode = False
        self.show_landmarks = False
        self.show_guide = False

        # Unified Hand Tracker Setup
        logger.info("Initializing Hand Tracker pipeline (max_num_hands=2)...")
        self.hand_tracker = UnifiedHandTracker()
        if self.hand_tracker.detector is None and not self.demo_mode:
            logger.error("No hand tracker available. Falling back to Demo Mode.")
            self.demo_mode = True

        # Pipelines
        self.smoother = TemporalSmoother()
        self.demo_gen = SyntheticDemoGenerator(width=self.target_w, height=self.target_h)

        # Camera handle
        self.cap: Optional[cv2.VideoCapture] = None

    def set_toast(self, message: str, duration: float = 2.5) -> None:
        """Schedules a temporary floating HUD toast message."""
        self.toast_message = message
        self.toast_expiry = time.perf_counter() + duration

    @classmethod
    def get_preferred_backend(cls) -> int:
        """Returns the optimal platform-native video capture backend."""
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        elif sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    @staticmethod
    def backend_name(backend: int) -> str:
        names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_AVFOUNDATION: "AVFOUNDATION", cv2.CAP_V4L2: "V4L2"}
        return names.get(backend, "ANY")

    @classmethod
    def discover_available_cameras(cls, max_probe: int = 6) -> List[Dict[str, Any]]:
        """
        Probes connected hardware camera indices and returns a list of active devices
        with index, default resolution, and backend details.
        """
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

        discovered: List[Dict[str, Any]] = []
        primary_backend = cls.get_preferred_backend()
        backends = [primary_backend]
        if primary_backend != cv2.CAP_ANY:
            backends.append(cv2.CAP_ANY)

        for idx in range(max_probe):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            backend_name = cls.backend_name(backend)
                            discovered.append(
                                {
                                    "index": idx,
                                    "width": w,
                                    "height": h,
                                    "fps": fps if fps > 0 else 30.0,
                                    "backend": backend_name,
                                }
                            )
                            cap.release()
                            break
                        cap.release()
                except Exception as e:
                    logger.debug("Probe failed for camera %d with backend %s: %s", idx, backend, e)

        return discovered

    def initialize_camera(self, target_cam_id: Optional[int] = None) -> bool:
        """
        Initializes webcam device with target resolution and platform-optimized backend.
        If target_cam_id is specified, attempts that index first; otherwise uses self.camera_id
        or falls back to any available discovered device.
        """
        if self.demo_mode:
            logger.info("Running in Demo Mode (--demo). Camera capture bypassed.")
            return True

        if target_cam_id is not None:
            self.camera_id = target_cam_id

        # Discover devices if list not yet cached
        if not self.available_cameras:
            discovered_info = self.discover_available_cameras(max_probe=6)
            self.available_cameras = [cam["index"] for cam in discovered_info]
            logger.info("Discovered %d active camera device(s): %s", len(self.available_cameras), self.available_cameras)

        # Build candidate indices starting with desired camera_id
        candidates = [self.camera_id] + [c for c in self.available_cameras if c != self.camera_id]
        if not candidates:
            candidates = [0, 1]

        # Release existing capture handle if open
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        primary_backend = self.get_preferred_backend()
        backends = [primary_backend]
        if primary_backend != cv2.CAP_ANY:
            backends.append(cv2.CAP_ANY)

        for idx in candidates:
            for backend in backends:
                backend_name = self.backend_name(backend)
                try:
                    logger.info("Attempting camera index %d with %s backend...", idx, backend_name)
                    cap = cv2.VideoCapture(idx, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        # Set target capture resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cap = cap
                            self.camera_id = idx
                            if idx not in self.available_cameras:
                                self.available_cameras.append(idx)
                                self.available_cameras.sort()
                            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            logger.info(
                                "✓ Camera index %d successfully acquired at %dx%d (%s)!",
                                idx,
                                actual_w,
                                actual_h,
                                backend_name,
                            )
                            return True
                        cap.release()
                except Exception as e:
                    logger.warning("Error probing camera index %d (%s): %s", idx, backend_name, str(e))

        logger.error("No functional webcam device could be opened.")
        return False

    def switch_camera(self, target_cam_id: Optional[int] = None) -> bool:
        """
        Switches between connected cameras dynamically during runtime.
        If target_cam_id is provided, switches directly to that camera index;
        otherwise cycles to the next available camera in self.available_cameras.
        """
        if not self.available_cameras:
            discovered = self.discover_available_cameras(max_probe=6)
            self.available_cameras = [c["index"] for c in discovered]

        if not self.available_cameras:
            self.set_toast("No camera devices detected!", duration=3.0)
            return False

        current_id = self.camera_id
        if target_cam_id is not None:
            next_cam = target_cam_id
        else:
            if current_id in self.available_cameras:
                curr_pos = self.available_cameras.index(current_id)
                next_cam = self.available_cameras[(curr_pos + 1) % len(self.available_cameras)]
            else:
                next_cam = self.available_cameras[0]

        if next_cam == current_id and self.cap is not None and not self.demo_mode:
            self.set_toast(f"Already on Cam #{current_id} (only 1 camera found)", duration=2.5)
            return True

        old_demo = self.demo_mode
        self.demo_mode = False
        success = self.initialize_camera(target_cam_id=next_cam)
        if success and self.cap is not None:
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_pos = self.available_cameras.index(self.camera_id) + 1 if self.camera_id in self.available_cameras else 1
            total = len(self.available_cameras)
            self.set_toast(f"Switched to Cam #{self.camera_id} ({actual_w}x{actual_h}) [{cam_pos}/{total}]", duration=3.0)
            logger.info("Successfully switched to Camera #%d", self.camera_id)
            return True
        else:
            logger.warning("Failed to switch to Camera #%d, reverting...", next_cam)
            self.demo_mode = old_demo
            self.initialize_camera(target_cam_id=current_id)
            self.set_toast(f"Camera #{next_cam} unavailable", duration=3.0)
            return False

    def rescan_cameras(self) -> None:
        """Rescans the system for newly connected or disconnected cameras."""
        self.set_toast("Scanning camera devices...", duration=1.5)
        discovered = self.discover_available_cameras(max_probe=6)
        self.available_cameras = [c["index"] for c in discovered]
        self.set_toast(f"Found {len(self.available_cameras)} camera(s): {self.available_cameras}", duration=3.0)
        logger.info("Rescanned cameras: %s", self.available_cameras)

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
        failed_reads = 0
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
                    # Throttle synthetic demo mode to target ~30 FPS to prevent 100% CPU lock
                    elapsed = time.perf_counter() - curr_time
                    target_interval = 1.0 / 30.0
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
                else:
                    if self.cap is None:
                        logger.warning("No camera handle available. Falling back to demo mode.")
                        self.demo_mode = True
                        self.set_toast("Camera lost. Switched to Demo Mode.", duration=3.0)
                        continue

                    success, frame = self.cap.read()
                    if not success or frame is None:
                        failed_reads += 1
                        if failed_reads >= 30:
                            logger.error("Camera stopped delivering frames. Falling back to Demo Mode.")
                            self.cap.release()
                            self.cap = None
                            self.demo_mode = True
                            failed_reads = 0  # a later successful camera switch starts clean
                        else:
                            cv2.waitKey(10)  # keep the window responsive while retrying
                        continue
                    failed_reads = 0

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
                active_toast = None
                if self.toast_message:
                    if time.perf_counter() < self.toast_expiry:
                        active_toast = self.toast_message
                    else:
                        self.toast_message = None

                cam_label = "SYNTHETIC DEMO" if self.demo_mode else f"CAM #{self.camera_id}"
                frame = HUDOverlay.draw_hud(
                    frame=frame,
                    pose=smoothed_pose,
                    fps=fps_tracker,
                    shadow_screen_mode=self.shadow_screen_mode,
                    show_landmarks=self.show_landmarks,
                    show_guide=self.show_guide,
                    is_live_camera=(not self.demo_mode),
                    camera_info=cam_label,
                    toast_message=active_toast,
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
                    self.set_toast(f"Theater Mode: {'ON' if self.shadow_screen_mode else 'OFF'}")
                    logger.info("Shadow Screen Mode toggled: %s", self.shadow_screen_mode)
                elif key in (ord("l"), ord("L")):
                    self.show_landmarks = not self.show_landmarks
                    self.set_toast(f"Hand Mesh: {'ON' if self.show_landmarks else 'OFF'}")
                    logger.info("Landmarks Overlay toggled: %s", self.show_landmarks)
                elif key in (ord("h"), ord("H")):
                    self.show_guide = not self.show_guide
                    logger.info("Gesture Cheat Sheet toggled: %s", self.show_guide)
                elif key in (ord("c"), ord("C")):
                    self.mirror_camera = not self.mirror_camera
                    self.set_toast(f"Mirroring: {'ON' if self.mirror_camera else 'OFF'}")
                    logger.info("Mirror Camera toggled: %s", self.mirror_camera)
                elif key in (ord("n"), ord("N"), 9):  # 'N' or TAB to cycle next camera
                    self.switch_camera()
                elif ord("0") <= key <= ord("9"):  # 0-9 to switch directly to camera index
                    self.switch_camera(target_cam_id=(key - ord("0")))
                elif key in (ord("r"), ord("R")):  # Rescan cameras
                    self.rescan_cameras()

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
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and list all detected camera devices on the system, then exit",
    )

    args = parser.parse_args()

    if args.list_cameras:
        print("=" * 64)
        print(" Hand-Gesture Shadow Puppets AI - Camera Device Discovery")
        print("=" * 64)
        print("Probing available video capture devices on system...")
        cams = ShadowPuppetsAI.discover_available_cameras(max_probe=8)
        if not cams:
            print("No video capture devices were detected.")
            print("Tip: Ensure your webcam is plugged in and camera permissions are granted.")
        else:
            print(f"Found {len(cams)} available camera device(s):\n")
            for c in cams:
                print(f"  [Camera #{c['index']}]")
                print(f"    Resolution: {c['width']}x{c['height']}")
                print(f"    Backend:    {c['backend']}")
                print(f"    FPS:        {c['fps']:.1f}")
                print(f"    Launch CLI: python shadow_puppets.py --camera {c['index']}\n")
            print("Runtime Keyboard Shortcuts:")
            print("  [N] or [TAB]  Cycle to the next detected camera")
            print("  [0-9]         Switch directly to camera #0 through #9")
            print("  [R]           Rescan for newly connected or disconnected cameras\n")
        return

    app = ShadowPuppetsAI(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        demo_mode=args.demo,
    )
    app.run()


if __name__ == "__main__":
    main()
