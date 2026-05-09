#!/usr/bin/env python3
"""Generate the cheat sheet PNG for abstrakt-deck.

Usage: python3 tools/generate_cheat_sheet.py
Requires: pip install pillow
Outputs: assets/cheat_sheet.png
"""

from PIL import Image, ImageDraw, ImageFont
import os

WIDTH = 800
HEIGHT = 760
BG = (0, 0, 0, 220)        # dark translucent background
HEADER = (255, 220, 80)     # gold for category headers
KEY = (120, 220, 255)       # cyan for key labels
TEXT = (240, 240, 240)      # near-white for descriptions

LINES = [
    ("HEADER", "ABSTRAKT-DECK CONTROLS"),
    ("BLANK", ""),
    ("CATEGORY", "GEOMETRY"),
    ("ENTRY", ("Shift+Tab", "cycle 3D shape (Cylinder/Sphere/Cube/Tetrahedron)")),
    ("ENTRY", ("P", "cycle painter (HueStripe/Spiral/Plasma)")),
    ("ENTRY", ("[ ]", "fold count (2-24)")),
    ("ENTRY", ("Z X", "kaleido zoom (0.3-1.5)")),
    ("ENTRY", (", .", "rotation speed (0-4x)")),
    ("BLANK", ""),
    ("CATEGORY", "FRAME"),
    ("ENTRY", ("1-7", "frame shape (None/Circle/Square/Round/Hex/Oct/Star)")),
    ("ENTRY", ("- =", "frame size")),
    ("ENTRY", ("R G B", "shift frame color hue")),
    ("BLANK", ""),
    ("CATEGORY", "EFFECTS"),
    ("ENTRY", ("I", "color invert")),
    ("ENTRY", ("T", "colorize tint")),
    ("ENTRY", (";", "colorize hue (+30°)")),
    ("ENTRY", ("9 0", "colorize intensity")),
    ("ENTRY", ("D", "distortion toggle")),
    ("ENTRY", ("Q W", "distortion amplitude")),
    ("ENTRY", ("E F", "distortion frequency")),
    ("BLANK", ""),
    ("CATEGORY", "AUDIO"),
    ("ENTRY", ("/ '", "bass-zoom intensity")),
    ("ENTRY", ("Space", "beat-shake toggle")),
    ("BLANK", ""),
    ("CATEGORY", "WINDOW & RECORDING"),
    ("ENTRY", ("F11", "fullscreen toggle")),
    ("ENTRY", ("F12", "video recording toggle")),
    ("BLANK", ""),
    ("CATEGORY", "PRESETS"),
    ("ENTRY", ("Ctrl+S", "save preset")),
    ("ENTRY", ("Ctrl+L", "load preset")),
    ("BLANK", ""),
    ("ENTRY", ("?", "toggle this help")),
    ("ENTRY", ("Esc", "exit")),
]


def find_mono_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    font_path = find_mono_font()
    if not font_path:
        raise RuntimeError(
            "No monospace font found — install dejavu-fonts-ttf or edit script"
        )

    font_header = ImageFont.truetype(font_path, 22)
    font_category = ImageFont.truetype(font_path, 18)
    font_entry = ImageFont.truetype(font_path, 16)

    img = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    x_key = 30
    x_desc = 130
    y = 20

    for kind, content in LINES:
        if kind == "HEADER":
            draw.text((x_key, y), content, fill=HEADER, font=font_header)
            y += 30
        elif kind == "CATEGORY":
            draw.text((x_key, y), content, fill=HEADER, font=font_category)
            y += 24
        elif kind == "ENTRY":
            key, desc = content
            draw.text((x_key, y), key, fill=KEY, font=font_entry)
            draw.text((x_desc, y), desc, fill=TEXT, font=font_entry)
            y += 20
        else:  # BLANK
            y += 10

    os.makedirs("assets", exist_ok=True)
    out = "assets/cheat_sheet.png"
    img.save(out)
    print(f"Wrote {out} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
