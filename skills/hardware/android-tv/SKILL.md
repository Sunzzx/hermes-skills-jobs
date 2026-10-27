---
name: android-tv-control
description: "Control Android TV (Basement TV) directly via ADB: launch streaming apps, control playback, volume, type text, and cast video URLs."
version: 1.0.0
author: Antigravity & Hermes
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Hardware, AndroidTV, ADB, Streaming, Remote]
    related_skills: []
---

# Android TV Control Skill

## When to Use
Use when the operator asks to control the TV, play media, open streaming apps (YouTube, SonyLIV, Cloudstream, Crunchyroll, Hotstar, Apple TV, JioTV+), control volume/playback, type text or search queries on TV, or cast/stream a URL to the TV screen.

## Capabilities & Direct CLI
All TV commands are available via the `tv` CLI tool (`~/.local/bin/tv`):

### 1. Launch Apps
```bash
tv app youtube        # Opens YouTube
tv app sonyliv        # Opens SonyLIV
tv app cloudstream    # Opens Cloudstream
tv app crunchyroll    # Opens Crunchyroll
tv app hotstar        # Opens Hotstar
tv app appletv        # Opens Apple TV
tv app jiotv          # Opens JioTV+
tv app settings       # Opens Android TV Settings
```

### 2. Media & Playback Controls
```bash
tv key play_pause     # Toggle play / pause
tv key play           # Resume playback
tv key pause          # Pause playback
tv key volume_up      # Increase volume
tv key volume_down    # Decrease volume
tv key mute           # Mute / unmute
tv key enter          # Select / OK
tv key back           # Go back
tv key home           # Go to home screen
tv key power          # Toggle power
tv key up / down / left / right
```

### 3. Open Video or Direct URL
```bash
tv url "https://www.youtube.com/watch?v=..."
```

### 4. Type Text & Search Queries
```bash
tv type "Search query or password"
```

### 5. Take Screen Capture
```bash
tv shot               # Saves screenshot to /tmp/tv_screen.png
```

### 6. Dynamic Auto-Discovery
The `tv` CLI automatically monitors and connects to the TV across any Wi-Fi network using mDNS resolution (`f9414852-ee4f-88ef-2193-6626589f407d.local.`).
