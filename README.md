# Hand-Gesture Shadow Puppets AI

A real-time Computer Vision & Interactive Graphics application built with **Python 3.10+**, **OpenCV**, **MediaPipe**, and **NumPy**. It tracks 21 3D hand landmarks in real-time at 30+ FPS, evaluates scale-invariant geometric heuristics to classify classic shadow puppet postures, and overlays dynamically tracked, stylized 2D vector silhouette characters directly onto the video stream.

---

## Architecture & Features

### 1. Zero Loose Pixel-Matching (Geometric Heuristics)
Recognition does not depend on brittle pixel masks or background subtraction. Instead, the engine extracts scale-invariant geometric relationships normalized by the distance between the **Wrist** and the **Middle Finger MCP**:
$$d_{norm}(p_1, p_2) = \frac{\|p_1 - p_2\|}{\|WRIST - MIDDLE\_MCP\|}$$

Five classic shadow puppet characters are recognized:
- 🦅 **Bird / Eagle** (Two-Handed or Single-Handed):
  - Two hands cross wrists or overlap thumbs with fingers spread outward like wings.
  - Left and right wing flight feathers dynamically flap in real-time as the user tilts their hands!
- 🐕 **Dog / Barking Wolf**:
  - Index finger erect for ear, middle and ring fingers extended together forward forming the upper snout.
  - Moving the thumb up and down dynamically pivots the lower jaw so the dog barks in real-time!
- 🦌 **Deer / Regal Stag**:
  - Index and pinky fingers raised high for antler horns; middle and ring fingertips touching thumb tip to form the tapered muzzle.
  - Procedural multi-tined branching antler rack with glowing eye cutout.
- 🐇 **Rabbit / Bunny**:
  - Index and middle fingers extended parallel upright (bunny ears), ring and pinky curled into palm, thumb folded over ring finger.
- 🦢 **Swan / Goose**:
  - Forearm held vertically (graceful neck), thumb pinched with index and middle fingertips to form the tapering bill/beak.

---

### 2. Aesthetic Procedural Silhouettes
- **Lightweight 2D Affine Transformation**: Polygons are defined in canonical coordinates and transformed via 2D rotation-scaling matrices ($M = s R$) using vectorized NumPy dot products in $<0.1$ ms.
- **Authentic Shadow Theater Styling**: Deep charcoal leather tone (`#1A1818`), subtle warm amber rim diffraction highlights (`#D7782D`), and illuminated eye apertures.
- **Shadow Screen Mode (`S` key)**: Transforms the video feed into an authentic backlit parchment screen where hands cast dramatic shadows.

---

### 3. Temporal State Machine & Anti-Jitter
- **Exponential Moving Average (EMA)**: Smooths translation ($x, y$), scaling ($s$), and rotation ($\theta$).
- **Continuous Angular Unwrapping**: Decomposes angles into $\cos(\theta)$ and $\sin(\theta)$ before smoothing to eliminate $360^\circ$ wraparound discontinuities.
- **State Debouncing**: Requires $N$ consecutive candidate frames before switching states, eliminating spurious flickering.
- **Cross-Fade Alpha Morphing**: Interpolates opacity smoothly ($0.0 \to 1.0$) across 12 frames when transitioning between puppets.

---

## Installation

Ensure you have Python 3.10+ installed.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy
```

---

## Running the Application

### 🚀 Automatic Launch Options (Zero Manual Setup)

1. **One-Click Finder Launcher**:
   - Double-click **`Start_Shadow_Puppets.command`** in Finder.
   - It automatically checks the virtual environment, verifies dependencies, and opens the app!

2. **Native macOS Application (`ShadowPuppets.app`)**:
   - Double-click **`ShadowPuppets.app`** or drag it into your macOS **Dock** or **`/Applications`** folder.
   - Includes standard macOS camera permission handling (`NSCameraUsageDescription`).

3. **Auto-Start on macOS Login (LaunchAgent)**:
   - To have Shadow Puppets automatically start every time you log into your Mac:
     ```bash
     ./setup_autostart.sh
     ```
   - To disable auto-start anytime:
     ```bash
     ./uninstall_autostart.sh
     ```

---

### Manual Terminal Execution
```bash
python shadow_puppets.py
```

### Select Specific Camera Device
```bash
python shadow_puppets.py --camera 1
```

### Synthetic Demo Mode (Camera-less / Automated Test)
If running on a system without a webcam, you can run the built-in procedural animation generator:
```bash
python shadow_puppets.py --demo
```

---

## Interactive Controls

| Hotkey | Action |
|:---:|:---|
| `H` | Toggle **Shadow Puppet Master Guide / Cheat Sheet** modal overlay |
| `S` | Toggle **Shadow Theater Mode** (Backlit parchment screen vs Webcam feed) |
| `L` | Toggle **MediaPipe Skeletal Landmark Mesh** visualization |
| `C` | Toggle **Camera Mirroring** (Left/Right flip) |
| `Q` or `ESC` | Cleanly exit application |

---

## Running Tests

Run the full unit and integration test suite:
```bash
python -m unittest test_shadow_puppets.py -v
```
