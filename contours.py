"""Inspect topology in a grayscale image. All areas and thresholds are in pixels."""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def fixture(name):
    """Draw simple, documented shapes in a reproducible 96 by 64 image."""
    img = np.zeros((64, 96), dtype=np.uint8)
    if name == "tray":
        img[7:38, 8:39] = 220           # part A: outer 31 by 31
        img[15:30, 16:31] = 0          # part A: interior hole
        img[10:29, 59:82] = 220        # part B: solid
        img[43:60, 44:67] = 220        # part C: hole with a loose island
        img[47:56, 49:62] = 0
        img[50:53, 54:57] = 220
        img[58, 8] = 220               # isolated speck
    elif name == "dark-tray":
        img[7:38, 8:39] = 220
        img[15:30, 16:31] = 0
        img[10:29, 59:82] = 220
        img = 255 - img
    elif name == "touching":
        img[11:45, 8:39] = 220
        img[20:33, 16:31] = 0
        img[11:45, 58:88] = 220
        img[27:30, 39:58] = 220       # bridge: one connected silhouette
    else:
        raise ValueError(f"unknown fixture: {name}")
    return img


def inspect(image, threshold=127, dark=False, min_outer=100.0, min_hole=40.0):
    """Return a JSON-ready report and the binary mask; input remains unchanged."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 2 or not image.size):
        raise ValueError("expected a nonempty two-dimensional uint8 grayscale image")
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError("threshold must be an integer from 0 to 255")
    if any(not np.isfinite(x) or x <= 0 for x in (min_outer, min_hole)):
        raise ValueError("minimum contour areas must be positive and finite")
    mode = cv.THRESH_BINARY_INV if dark else cv.THRESH_BINARY
    _, mask = cv.threshold(image, threshold, 255, mode)
    boundaries, links = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    external, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    parents = [] if links is None else links[0, :, 3].tolist()
    # Contour order is not part of the public interface. Compute depth from parents.
    depths = {}
    for index in range(len(boundaries)):
        unresolved = []
        cursor = index
        while cursor != -1 and cursor not in depths:
            unresolved.append(cursor)
            cursor = parents[cursor]
        depth = depths.get(cursor, -1)
        for node in reversed(unresolved):
            depth += 1
            depths[node] = depth
    children = [[] for _ in boundaries]
    for index, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(index)

    areas = [cv.contourArea(boundary) for boundary in boundaries]
    parts = []
    for i, contour in enumerate(boundaries):
        if depths[i] % 2 or areas[i] < min_outer:
            continue
        x, y, w, h = cv.boundingRect(contour)
        holes = [j for j in children[i] if areas[j] >= min_hole]
        parts.append({
            "box_xywh_px": [x, y, w, h],
            "touches_image_border": x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0],
            "depth": depths[i],
            "outer_area_px2": areas[i],
            "direct_holes": len(children[i]),
            "qualifying_holes": len(holes),
            "qualifying_hole_areas_px2": sorted(areas[j] for j in holes),
            "classification": {0: "no-qualifying-hole", 1: "one-qualifying-hole"}.get(
                len(holes), "multiple-qualifying-holes"
            ),
        })
    parts.sort(key=lambda p: (p["box_xywh_px"][0], p["box_xywh_px"][1], p["depth"]))
    report = {
        "size_wh_px": [mask.shape[1], mask.shape[0]],
        "foreground_px": int(np.count_nonzero(mask)),
        "external_boundaries": len(external),
        "tree_boundaries": len(boundaries),
        "foreground_boundaries_before_filter": sum(d % 2 == 0 for d in depths.values()),
        "parts": parts,
    }
    return report, mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", choices=("tray", "dark-tray", "touching"))
    source.add_argument("--image", type=Path, help="8-bit grayscale PNG")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--dark", action="store_true", help="pixels at or below the threshold are foreground")
    parser.add_argument("--min-outer", type=float, default=100)
    parser.add_argument("--min-hole", type=float, default=40)
    parser.add_argument("--mask-out", type=Path, help="save the binary mask as a PNG")
    args = parser.parse_args(argv)
    if args.mask_out and args.mask_out.suffix.lower() != ".png":
        parser.error("--mask-out must have a .png extension")
    if args.image and args.mask_out and args.image.resolve() == args.mask_out.resolve():
        parser.error("mask output must not overwrite the input image")
    try:
        image = fixture(args.fixture) if args.fixture else cv.imdecode(
            np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv.IMREAD_UNCHANGED
        )
        report, mask = inspect(image, args.threshold, args.dark, args.min_outer, args.min_hole)
        if args.mask_out:
            ok, png = cv.imencode(".png", mask)
            if not ok:
                raise ValueError("could not encode mask")
            with args.mask_out.open("xb") as output:
                output.write(png.tobytes())
    except (OSError, ValueError, cv.error) as err:
        parser.error(str(err))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
