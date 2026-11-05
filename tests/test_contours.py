from collections import deque
import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import cv2 as cv
import numpy as np

from contours import fixture, inspect, main


def component_counts(mask):
    """Independent pixel flood fill: eight neighbors for parts, four for background."""
    rows, cols = mask.shape
    seen = set()
    objects, holes = 0, 0
    for y in range(rows):
        for x in range(cols):
            if (y, x) in seen:
                continue
            foreground = bool(mask[y, x])
            seen.add((y, x))
            pending = deque([(y, x)])
            border = False
            while pending:
                yy, xx = pending.popleft()
                border |= yy in (0, rows - 1) or xx in (0, cols - 1)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    if not foreground and dy and dx:
                        continue
                    a, b = yy + dy, xx + dx
                    if (0 <= a < rows and 0 <= b < cols and (a, b) not in seen
                            and bool(mask[a, b]) == foreground):
                        seen.add((a, b))
                        pending.append((a, b))
            if foreground:
                objects += 1
            elif not border:
                holes += 1
    return objects, holes


class ContoursTest(unittest.TestCase):
    def test_tree_counts_agree_with_independent_flood_fill_on_120_cases(self):
        rng = np.random.default_rng(151)
        for _ in range(120):
            mask = np.zeros((40, 48), np.uint8)
            for y, x in ((2, 2), (2, 25), (22, 2), (22, 25)):
                shape = rng.integers(4)
                if shape:
                    w, h = rng.integers(11, 16, size=2)
                    mask[y:y+h, x:x+w] = 255
                    if shape >= 2:
                        mask[y+2:y+h-2, x+2:x+w-2] = 0
                    if shape == 3:
                        mask[y+4:y+7, x+4:x+7] = 255
            expected_objects, expected_holes = component_counts(mask)
            actual, _ = inspect(mask, min_outer=1, min_hole=1)
            self.assertEqual(actual["foreground_boundaries_before_filter"], expected_objects)
            self.assertEqual(actual["tree_boundaries"] - expected_objects, expected_holes)
            self.assertEqual(sum(p["direct_holes"] for p in actual["parts"]), expected_holes)

    def test_rectangle_polygon_vs_pixel_area(self):
        img = np.zeros((30, 40), np.uint8)
        img[5:12, 7:18] = 255
        report, _ = inspect(img, min_outer=1)
        self.assertEqual(report["foreground_px"], 77)
        self.assertEqual(report["parts"][0]["outer_area_px2"], 60)
        self.assertEqual(report["parts"][0]["box_xywh_px"], [7, 5, 11, 7])

    def test_island_is_foreground_not_a_second_hole(self):
        report, _ = inspect(fixture("tray"), min_outer=1)
        by_box = {tuple(p["box_xywh_px"]): p for p in report["parts"]}
        self.assertEqual(report["external_boundaries"], 4)
        self.assertEqual(report["foreground_boundaries_before_filter"], 5)
        self.assertEqual(by_box[54, 50, 3, 3]["depth"], 2)
        self.assertEqual(by_box[54, 50, 3, 3]["direct_holes"], 0)
        self.assertEqual(by_box[44, 43, 23, 17]["direct_holes"], 1)

    def test_polarity_and_threshold_equality(self):
        r, mask = inspect(np.array([[126, 127, 128]], np.uint8))
        self.assertEqual(mask.tolist(), [[0, 0, 255]])
        _, mask = inspect(np.array([[126, 127, 128]], np.uint8), dark=True)
        self.assertEqual(mask.tolist(), [[255, 255, 0]])
        good, _ = inspect(fixture("dark-tray"), dark=True)
        bad, _ = inspect(fixture("dark-tray"))
        self.assertEqual(len(good["parts"]), 2)
        self.assertTrue(bad["parts"][0]["touches_image_border"])

    def test_filter_preserves_direct_hole_count_and_reports_multiple(self):
        img = np.zeros((40, 60), np.uint8)
        img[3:36, 3:56] = 255
        img[8:18, 8:20] = 0
        img[20:32, 25:48] = 0
        for gate, count in ((1, 2), (200, 1), (1000, 0)):
            with self.subTest(gate=gate):
                r, _ = inspect(img, min_hole=gate)
                self.assertEqual(r["parts"][0]["direct_holes"], 2)
                self.assertEqual(r["parts"][0]["qualifying_holes"], count)
                if count == 2:
                    self.assertEqual(r["parts"][0]["classification"], "multiple-qualifying-holes")

    def test_touching_parts_remain_one_silhouette(self):
        img = fixture("touching")
        r, _ = inspect(img)
        self.assertEqual(len(r["parts"]), 1)
        self.assertEqual(r["parts"][0]["qualifying_holes"], 1)
        img[27:30, 39:58] = 0
        r, _ = inspect(img)
        self.assertEqual(len(r["parts"]), 2)

    def test_empty_mask_and_input_untouched(self):
        r, _ = inspect(np.zeros((3, 2), np.uint8))
        self.assertEqual(r["parts"], [])
        self.assertEqual(r["tree_boundaries"], 0)
        image = fixture("tray")
        before = image.copy()
        inspect(image)
        np.testing.assert_array_equal(image, before)

    def test_invalid_image_threshold_and_area(self):
        for image in (None, np.zeros((3, 2, 3), np.uint8), np.zeros((1, 2), np.uint16)):
            with self.subTest(image_type=type(image)), self.assertRaises(ValueError):
                inspect(image)
        for kwargs in ({"threshold": -1}, {"threshold": 256}, {"min_outer": 0},
                       {"min_hole": float("nan")}, {"min_hole": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inspect(fixture("tray"), **kwargs)

    def test_cli_round_trip_png_and_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "space name.png"
            mask = Path(d) / "mask.png"
            self.assertTrue(cv.imwrite(str(file), fixture("tray")))
            args = ["--image", str(file), "--mask-out", str(mask)]
            with contextlib.redirect_stdout(StringIO()) as output:
                self.assertEqual(main(args), 0)
            self.assertEqual(len(json.loads(output.getvalue())["parts"]), 3)
            np.testing.assert_array_equal(cv.imread(str(mask), cv.IMREAD_UNCHANGED),
                                          inspect(fixture("tray"))[1])
            with contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit) as err:
                main(args)
            self.assertEqual(err.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
