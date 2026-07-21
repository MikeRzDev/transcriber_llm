#!/usr/bin/env bash
# Package the compiled release binary as "Transcribe Speech.app" and install it
# into /Applications so it shows up in Launchpad/Spotlight. Launching it
# opens the TUI in a Terminal window. The binary is copied into the bundle —
# re-run this script after rebuilding to refresh it. (bash 3.2 compatible)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/transcribe-stt"
APP_NAME="Transcribe Speech"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: app bundles are a macOS thing" >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "→ no release binary yet — building…"
  (cd "$ROOT" && cargo build --release)
fi

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
VERSION="${VERSION:-0.1.0}"

DEST="/Applications"
[[ -w "$DEST" ]] || DEST="$HOME/Applications"
mkdir -p "$DEST"
APP="$DEST/$APP_NAME.app"

echo "→ assembling $APP (v$VERSION)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BIN" "$APP/Contents/MacOS/transcribe-stt-bin"

# The bundle's executable is a tiny launcher that hands the TUI to Terminal
cat > "$APP/Contents/MacOS/launcher" <<'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec open -a Terminal "$DIR/transcribe-stt-bin"
LAUNCHER
chmod +x "$APP/Contents/MacOS/launcher"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>local.transcribe-stt</string>
  <key>CFBundleVersion</key>
  <string>$VERSION</string>
  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleExecutable</key>
  <string>launcher</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
</dict>
</plist>
PLIST

# Ad-hoc signing keeps Gatekeeper happy for a locally built bundle
codesign --force -s - "$APP/Contents/MacOS/transcribe-stt-bin" 2>/dev/null || true
codesign --force -s - "$APP" 2>/dev/null || true

# Nudge LaunchServices so Launchpad/Spotlight pick it up right away
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" >/dev/null 2>&1 || true

echo "done: $APP"
echo "find '$APP_NAME' in Launchpad/Spotlight — it opens transcribe-stt in Terminal"
