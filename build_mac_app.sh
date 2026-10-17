#!/usr/bin/env bash
# ==============================================================================
# Build Native macOS Application Bundle (ShadowPuppets.app)
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
APP_NAME="ShadowPuppets.app"
APP_DIR="$SCRIPT_DIR/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Building native macOS Application Bundle: $APP_NAME..."

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Write Info.plist with Camera Permissions
cat <<EOF > "$CONTENTS_DIR/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>ShadowPuppets</string>
    <key>CFBundleIdentifier</key>
    <string>com.dillandeluxe.shadowpuppets</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Shadow Puppets AI</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSCameraUsageDescription</key>
    <string>Shadow Puppets AI requires webcam access to track hand gestures in real-time.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Executable entrypoint
cat <<EOF > "$MACOS_DIR/ShadowPuppets"
#!/bin/bash
DIR="$SCRIPT_DIR"
cd "\$DIR"
exec "\$DIR/.venv/bin/python" "\$DIR/shadow_puppets.py"
EOF

chmod +x "$MACOS_DIR/ShadowPuppets"

echo "=================================================="
echo "✓ Native macOS App created at: $APP_DIR"
echo "  You can:"
echo "  1. Double-click ShadowPuppets.app to launch"
echo "  2. Drag it to your macOS Dock or /Applications"
echo "=================================================="
