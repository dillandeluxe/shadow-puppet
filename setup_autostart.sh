#!/usr/bin/env bash
# ==============================================================================
# Setup Auto-Start on macOS User Login (LaunchAgent)
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PLIST_NAME="com.dillandeluxe.shadowpuppets.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$TARGET_DIR/$PLIST_NAME"

mkdir -p "$TARGET_DIR"

cat <<EOF > "$TARGET_PLIST"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dillandeluxe.shadowpuppets</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/.venv/bin/python</string>
        <string>$SCRIPT_DIR/shadow_puppets.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/shadow_puppets.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/shadow_puppets.err</string>
</dict>
</plist>
EOF

echo "✓ LaunchAgent created at: $TARGET_PLIST"
launchctl unload "$TARGET_PLIST" 2>/dev/null || true
launchctl load "$TARGET_PLIST"

echo "=================================================="
echo "✓ Hand-Gesture Shadow Puppets AI will now start"
echo "  automatically every time you log in to your Mac!"
echo "  To disable auto-start, run: ./uninstall_autostart.sh"
echo "=================================================="
