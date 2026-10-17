#!/usr/bin/env bash
# ==============================================================================
# Disable Auto-Start on macOS User Login
# ==============================================================================

set -e

PLIST_NAME="com.dillandeluxe.shadowpuppets.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/$PLIST_NAME"

if [ -f "$TARGET_PLIST" ]; then
    launchctl unload "$TARGET_PLIST" 2>/dev/null || true
    rm -f "$TARGET_PLIST"
    echo "✓ Auto-start disabled. LaunchAgent removed from $TARGET_PLIST."
else
    echo "Notice: Auto-start LaunchAgent was not installed."
fi
