"""Procedural puppet silhouette renderer and HUD overlay."""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


from gestures import PuppetGesture, PuppetPose


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

        # Draw components with support for cross-fade morphing between gestures
        morph_curr = float(pose.morph_t)
        morph_prev = (1.0 - morph_curr) if (pose.prev_gesture != PuppetGesture.NONE and morph_curr < 1.0) else 0.0

        gesture_layers = []
        if morph_prev > 0.01:
            prev_polys, prev_eyes = cls._get_canonical_geometry(pose, gesture=pose.prev_gesture)
            if prev_polys:
                gesture_layers.append((prev_polys, prev_eyes, morph_prev))

        polygons, eye_circles = cls._get_canonical_geometry(pose)
        if polygons:
            gesture_layers.append((polygons, eye_circles, morph_curr if morph_prev > 0.01 else 1.0))

        if not gesture_layers:
            return output

        for polys, eyes, weight in gesture_layers:
            alpha_int = int(np.clip(255 * weight, 0, 255))
            for poly in polys:
                poly_arr = np.array(poly, dtype=np.float32)
                transformed = (poly_arr @ M.T) + anchor
                pts_int = np.int32(transformed)

                # Fill silhouette polygon
                cv2.fillPoly(overlay, [pts_int], cls.CHARCOAL_SHADOW, lineType=cv2.LINE_AA)
                cv2.fillPoly(mask, [pts_int], alpha_int, lineType=cv2.LINE_AA)

                # Soft rim-glow edge
                cv2.polylines(overlay, [pts_int], isClosed=True, color=cls.AMBER_RIM_GLOW, thickness=2, lineType=cv2.LINE_AA)

            # Transform and carve glowing eye apertures
            for center_u, center_v, radius in eyes:
                eye_pos = (np.array([center_u, center_v], dtype=np.float32) @ M.T) + anchor
                eye_r = max(2, int(radius * s))
                pt_center = (int(eye_pos[0]), int(eye_pos[1]))

                # Draw glowing aperture
                cv2.circle(overlay, pt_center, eye_r, cls.EYE_APERTURE_COLOR, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, pt_center, eye_r + 1, cls.AMBER_RIM_GLOW, thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(mask, pt_center, eye_r + 1, alpha_int, thickness=-1, lineType=cv2.LINE_AA)

        # Alpha composite overlay over frame with subpixel anti-aliasing preservation
        effective_alpha = float(np.clip(alpha * 0.88, 0.0, 1.0))
        active_indices = mask > 0
        if np.any(active_indices):
            alpha_f = (mask[active_indices].astype(np.float32) / 255.0) * effective_alpha
            alpha_3d = alpha_f[:, np.newaxis]
            fg = overlay[active_indices].astype(np.float32)
            bg = output[active_indices].astype(np.float32)
            output[active_indices] = np.uint8(fg * alpha_3d + bg * (1.0 - alpha_3d))

        return output

    @classmethod
    def _apply_parchment_screen(cls, frame: np.ndarray) -> np.ndarray:
        """
        Simulates an authentic warm backlit parchment screen, casting dramatic hand shadows.
        """
        # Convert to grayscale to preserve hand shadow contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optimized uint8 scaling without float64 heap allocations
        b = cv2.convertScaleAbs(gray, alpha=0.70, beta=35)
        g = cv2.convertScaleAbs(gray, alpha=0.85, beta=45)
        r = cv2.convertScaleAbs(gray, alpha=0.95, beta=50)
        parchment = cv2.merge([b, g, r])
        return cv2.addWeighted(parchment, 0.85, frame, 0.15, 0)

    @classmethod
    def _get_canonical_geometry(
        cls, pose: PuppetPose, gesture: Optional[PuppetGesture] = None
    ) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float, float]]]:
        """
        Returns procedural canonical 2D contours and eye apertures for each puppet.
        Coordinates are defined around (0, 0) anchor:
        +v is along the fingers/hand axis, +u is lateral.
        """
        active_gesture = gesture if gesture is not None else pose.gesture
        polygons: List[List[Tuple[float, float]]] = []
        eyes: List[Tuple[float, float, float]] = []

        if active_gesture == PuppetGesture.DOG:
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

        elif active_gesture == PuppetGesture.DEER:
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

        elif active_gesture == PuppetGesture.BIRD:
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
                # Rotate symmetrically: positive wing tilt lifts both wings upward (+v)
                du = u - (-22.0)
                dv = v - 35.0
                ru = -22.0 + (du * math.cos(left_rad) + dv * math.sin(left_rad))
                rv = 35.0 + (-du * math.sin(left_rad) + dv * math.cos(left_rad))
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

        elif active_gesture == PuppetGesture.RABBIT:
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

        elif active_gesture == PuppetGesture.SWAN:
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
        camera_info: str = "LIVE CAM",
        toast_message: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws the on-screen UI overlays over the frame.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # ----------------------------------------------------------------------
        # Top-Left: Active Puppet Status Badge
        # ----------------------------------------------------------------------
        cls._draw_status_card(canvas, pose, fps, shadow_screen_mode, show_landmarks, camera_info)

        # ----------------------------------------------------------------------
        # Floating Toast Banner Notification (if active)
        # ----------------------------------------------------------------------
        if toast_message:
            cls._draw_toast(canvas, w, h, toast_message)

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
        camera_info: str = "LIVE CAM",
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
        info_str = f"FPS: {fps:.1f}  |  {camera_info}  |  Mesh: {'ON' if landmarks_on else 'OFF'}"
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
    def _draw_toast(cls, canvas: np.ndarray, w: int, h: int, message: str) -> None:
        """Renders an elegant glassmorphism banner pill in the top-center of the screen."""
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.52
        thickness = 1
        text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
        tw, th = text_size
        pad_x, pad_y = 20, 10
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        bx = max(10, (w - bw) // 2)
        by = 20

        sub = canvas[by : by + bh, bx : bx + bw]
        card = np.full_like(sub, (20, 24, 34))
        cv2.addWeighted(card, 0.88, sub, 0.12, 0, sub)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), cls.COLOR_ACCENT, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            message,
            (bx + pad_x, by + th + pad_y - 2),
            font,
            font_scale,
            cls.COLOR_TEXT_MAIN,
            thickness,
            lineType=cv2.LINE_AA,
        )

    @classmethod
    def _draw_controls_bar(cls, canvas: np.ndarray, w: int, h: int, guide_open: bool) -> None:
        """Draws a clean bottom shortcut helper bar."""
        bar_text = "[H] Guide  |  [S] Theater  |  [L] Mesh  |  [C] Flip  |  [N] Switch Cam  |  [0-9] Cam #  |  [Q/ESC] Quit"
        cv2.putText(
            canvas,
            bar_text,
            (20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
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
        cv2.addWeighted(card, 0.72, sub, 0.28, 0, sub)
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
