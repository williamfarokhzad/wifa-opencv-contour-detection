"""Rebuild the original lesson figures from the real fixture masks.

From the project root: python -m pip install Pillow==13.0.0
                         python diagrams/render.py
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contours import fixture, inspect

OUT = Path(__file__).resolve().parent
BG = "#102b39"
INK = "#e7f5f7"
SUB = "#a5c6cc"
FG = "#bceee5"
MINT = "#63e5bb"
GOLD = "#ffd173"
PINK = "#ff9ab2"


def label(draw, xy, value, size, color=INK):
    draw.text(xy, value, font=ImageFont.load_default(size=size), fill=color)


def canvas(size, heading):
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    label(draw, (28, 21), heading, 29)
    return image, draw


def mask_panel(image, mask, origin, scale):
    tile = Image.frombytes("L", (mask.shape[1], mask.shape[0]), mask.tobytes())
    tile = tile.resize((tile.width * scale, tile.height * scale), Image.Resampling.NEAREST)
    block = Image.new("RGB", tile.size, "#081a24")
    block.paste(Image.new("RGB", tile.size, FG), mask=tile)
    image.paste(block, origin)


def save(image, name):
    image.save(OUT / name, "WEBP", lossless=True, method=6)


def tray():
    im, d = canvas((840, 510), "One tray, two kinds of boundary")
    report, mask = inspect(fixture("tray"))
    x, y, scale = 28, 82, 5
    mask_panel(im, mask, (x, y), scale)
    # A, B, and C use the bounding boxes from the real report.
    for part, letter in zip(report["parts"], "ACB", strict=True):
        a, b, w, h = part["box_xywh_px"]
        box = (x + a*scale-3, y + b*scale-3, x+(a+w)*scale+2, y+(b+h)*scale+2)
        d.rectangle(box, outline=MINT, width=2)
        label(d, (box[0], box[1]-24), letter, 21, MINT)
    for cx, cy, r, color in ((23, 22, 9, GOLD), (55, 51, 8, GOLD), (8, 58, 3, PINK)):
        d.ellipse((x+(cx-r)*scale,y+(cy-r)*scale,x+(cx+r)*scale,y+(cy+r)*scale), outline=color, width=3)
    for row, lines in enumerate((
        ("A  Ring", "One inner boundary"),
        ("B  Solid", "No inner boundary"),
        ("C  Ring + island", "One hole, one island inside"),
        ("Speck", "Below the outer area gate"),
    )):
        label(d, (534, 86 + row*76), lines[0], 24, PINK if row == 3 else MINT)
        label(d, (534, 115 + row*76), lines[1], 17, SUB)
    label(d, (28, 428), f"External boundaries: {report['external_boundaries']}", 22)
    label(d, (28, 464), f"Tree boundaries: {report['tree_boundaries']}  ·  Parts after area filter: {len(report['parts'])}", 22)
    save(im, "contour-tray.webp")


def polarity():
    im, d = canvas((850, 370), "Threshold polarity changes the answer")
    wrong, wrong_mask = inspect(fixture("dark-tray"))
    correct, correct_mask = inspect(fixture("dark-tray"), dark=True)
    for x, title, mask in ((28, "Default: bright foreground", wrong_mask),
                           (438, "Inverted: dark foreground", correct_mask)):
        label(d, (x, 69), title, 21)
        mask_panel(im, mask, (x, 108), 4)
    # Draw frame markers over the full-frame foreground from the default pass.
    p = wrong["parts"][0]
    assert p["touches_image_border"]
    d.rectangle((28,108,411,363), outline=PINK, width=4)
    assert all(not p["touches_image_border"] for p in correct["parts"])
    save(im, "contour-polarity.webp")


if __name__ == "__main__":
    tray()
    polarity()
