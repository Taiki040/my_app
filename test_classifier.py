import unittest

from classifier import classify_operator, is_open_palm, is_pinky_only


class OperatorClassificationTests(unittest.TestCase):
    def test_horizontal_single_stroke_is_minus(self) -> None:
        self.assertEqual(classify_operator([[(-0.5, 0), (0.5, 0)]]), "-")

    def test_two_parallel_horizontal_strokes_are_equals(self) -> None:
        strokes = [
            [(-0.5, -0.15), (0.5, -0.15)],
            [(-0.5, 0.15), (0.5, 0.15)],
        ]
        self.assertEqual(classify_operator(strokes), "=")

    def test_perpendicular_strokes_are_plus(self) -> None:
        strokes = [
            [(-0.4, 0), (0.4, 0)],
            [(0, -0.4), (0, 0.4)],
        ]
        self.assertEqual(classify_operator(strokes), "+")

    def test_three_stroke_plus_with_looser_aspect_ratio(self) -> None:
        strokes = [
            [(-0.35, 0), (0.35, 0)],
            [(0, -0.5), (0, 0.5)],
            [(-0.03, -0.03), (0.03, 0.03)],
        ]
        self.assertEqual(classify_operator(strokes), "+")

    def test_crossing_diagonal_strokes_are_multiply(self) -> None:
        strokes = [
            [(-0.4, -0.4), (0.4, 0.4)],
            [(-0.4, 0.4), (0.4, -0.4)],
        ]
        self.assertEqual(classify_operator(strokes), "*")

    def test_single_diagonal_stroke_is_divide(self) -> None:
        self.assertEqual(classify_operator([[(-0.4, 0.4), (0.4, -0.4)]]), "/")

    def test_horizontal_line_with_dots_is_divide(self) -> None:
        strokes = [
            [(-1, 0), (1, 0)],
            [(0, -0.2)],
            [(0, 0.2)],
        ]
        self.assertEqual(classify_operator(strokes), "/")

    def test_unmatched_stroke_falls_through_to_digit_classifier(self) -> None:
        strokes = [[(-0.2, -0.2), (0, 0.2), (0.2, -0.2)]]
        self.assertIsNone(classify_operator(strokes))


class OpenPalmTests(unittest.TestCase):
    def test_all_four_fingertips_above_pip_joints(self) -> None:
        landmarks = [(0.0, 0.5, 0.0) for _ in range(21)]
        for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
            landmarks[tip] = (0.0, 0.2, 0.0)
            landmarks[pip] = (0.0, 0.4, 0.0)
        self.assertTrue(is_open_palm(landmarks))

    def test_equal_or_lower_fingertip_does_not_count_as_open_palm(self) -> None:
        landmarks = [(0.0, 0.5, 0.0) for _ in range(21)]
        for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
            landmarks[tip] = (0.0, 0.2, 0.0)
            landmarks[pip] = (0.4, 0.4, 0.0)
        landmarks[20] = (0.0, 0.4, 0.0)
        self.assertFalse(is_open_palm(landmarks))


class PinkyOnlyTests(unittest.TestCase):
    def test_only_pinky_above_pip_joints(self) -> None:
        landmarks = [(0.0, 0.5, 0.0) for _ in range(21)]
        for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
            landmarks[tip] = (0.0, 0.6, 0.0)
            landmarks[pip] = (0.0, 0.4, 0.0)
        landmarks[20] = (0.0, 0.2, 0.0)
        self.assertTrue(is_pinky_only(landmarks))

    def test_other_extended_fingers_do_not_count(self) -> None:
        landmarks = [(0.0, 0.5, 0.0) for _ in range(21)]
        for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
            landmarks[tip] = (0.0, 0.6, 0.0)
            landmarks[pip] = (0.0, 0.4, 0.0)
        landmarks[20] = (0.0, 0.2, 0.0)
        landmarks[8] = (0.0, 0.2, 0.0)
        self.assertFalse(is_pinky_only(landmarks))


if __name__ == "__main__":
    unittest.main()
