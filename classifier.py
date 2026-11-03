"""MediaPipe hand-pose detection and air-writing gesture tracking."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Iterable, Sequence
import warnings

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DrawingMode(str, Enum):
    DRAWING = "DRAWING"
    ERASER = "ERASER"
    PAUSED = "PAUSED"
    CLEAR = "CLEAR"
    IDLE = "IDLE"


Landmark = tuple[float, float, float]
Stroke = list[list[tuple[float, float]]]

_FINGER_TIPS = (8, 12, 16, 20)
_FINGER_PIPS = (6, 10, 14, 18)
_ERASER_RADIUS_PX = 18.0


def is_open_palm(landmarks: Sequence[Landmark]) -> bool:
    """Return whether all four non-thumb fingertips are above their PIP joints."""
    return all(
        landmarks[tip][1] < landmarks[joint][1]
        for tip, joint in zip(_FINGER_TIPS, _FINGER_PIPS)
    )


def is_pinky_only(landmarks: Sequence[Landmark]) -> bool:
    """Return whether only the pinky is extended relative to its PIP joint."""
    fingers_extended = [
        landmarks[tip][1] < landmarks[joint][1]
        for tip, joint in zip(_FINGER_TIPS, _FINGER_PIPS)
    ]
    return fingers_extended == [False, False, False, True]


class SimpleCNN(nn.Module):
    """CNN for classifying 28x28 MNIST-style digit images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.max_pool2d(F.relu(self.conv1(image)), 2)
        image = F.max_pool2d(F.relu(self.conv2(image)), 2)
        image = image.flatten(start_dim=1)
        image = self.dropout(F.relu(self.fc1(image)))
        return self.fc2(image)


_MODEL_PATH = Path(__file__).with_name("digit_model.pth")
_digit_model: SimpleCNN | None = None
if _MODEL_PATH.is_file():
    _digit_model = SimpleCNN()
    _digit_model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu", weights_only=True))
    _digit_model.eval()
else:
    warnings.warn(
        f"Digit model not found at {_MODEL_PATH}. Run train.py to enable digit prediction.",
        RuntimeWarning,
        stacklevel=2,
    )


def _strokes_to_image(strokes: Stroke) -> np.ndarray:
    valid_strokes = [
        [(float(x), float(y)) for x, y in stroke if np.isfinite(x) and np.isfinite(y)]
        for stroke in strokes
    ]
    valid_strokes = [stroke for stroke in valid_strokes if stroke]
    points = [point for stroke in valid_strokes for point in stroke]
    if not points:
        raise ValueError("At least one finite stroke point is required.")

    coordinates = np.asarray(points, dtype=np.float32)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = 80.0 / float(max(span))
    center = (minimum + maximum) / 2
    mapped = (coordinates - center) * scale + 56

    image = np.zeros((112, 112), dtype=np.uint8)
    point_index = 0
    for stroke in valid_strokes:
        stroke_points = []
        for _ in stroke:
            x, y = mapped[point_index]
            stroke_points.append((int(round(x)), int(round(y))))
            point_index += 1
        if len(stroke_points) == 1:
            cv2.circle(image, stroke_points[0], 4, 255, thickness=-1, lineType=cv2.LINE_AA)
        elif len(stroke_points) > 1:
            for start, end in zip(stroke_points, stroke_points[1:]):
                cv2.line(image, start, end, 255, thickness=8, lineType=cv2.LINE_AA)

    return cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)


def classify_operator(strokes: Stroke) -> str | None:
    """Classify a drawing as an arithmetic operator, if it matches simple geometry rules."""
    valid_strokes = [
        np.asarray(
            [(float(x), float(y)) for x, y in stroke if np.isfinite(x) and np.isfinite(y)],
            dtype=np.float32,
        )
        for stroke in strokes
    ]
    valid_strokes = [stroke for stroke in valid_strokes if len(stroke)]
    if not valid_strokes:
        return None

    points = np.concatenate(valid_strokes)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    width, height = (maximum - minimum).tolist()
    aspect_ratio = width / max(height, 1e-6)

    centers = [(stroke.min(axis=0) + stroke.max(axis=0)) / 2 for stroke in valid_strokes]
    stroke_sizes = [stroke.max(axis=0) - stroke.min(axis=0) for stroke in valid_strokes]
    angles: list[float] = []
    for stroke in valid_strokes:
        if len(stroke) < 2:
            angles.append(0.0)
        else:
            centered_points = stroke - stroke.mean(axis=0)
            covariance = centered_points.T @ centered_points
            _, eigenvectors = np.linalg.eigh(covariance)
            direction = eigenvectors[:, -1]
            angles.append(float(np.degrees(np.arctan2(direction[1], direction[0]))) % 180)

    def is_horizontal(angle: float) -> bool:
        return min(angle, 180 - angle) <= 25

    def is_vertical(angle: float) -> bool:
        return abs(angle - 90) <= 25

    horizontal_lines = [
        is_horizontal(angle) and size[0] >= width * 0.4
        for angle, size in zip(angles, stroke_sizes)
    ]

    for first in range(len(valid_strokes)):
        for second in range(first + 1, len(valid_strokes)):
            if not horizontal_lines[first] or not horizontal_lines[second]:
                continue
            y_separation = abs(float(centers[first][1] - centers[second][1]))
            x_overlap = min(
                float(valid_strokes[first][:, 0].max()),
                float(valid_strokes[second][:, 0].max()),
            ) - max(
                float(valid_strokes[first][:, 0].min()),
                float(valid_strokes[second][:, 0].min()),
            )
            if y_separation > max(height * 0.1, 1e-6) and x_overlap >= width * 0.4:
                return "="

    if aspect_ratio > 1.8 and len(valid_strokes) >= 3:
        long_horizontal = [
            index for index, is_line in enumerate(horizontal_lines) if is_line
        ]
        small_strokes = [
            index
            for index, size in enumerate(stroke_sizes)
            if max(float(size[0]), float(size[1])) <= max(width, height) * 0.2
        ]
        if long_horizontal and len(small_strokes) >= 2:
            line_index = max(long_horizontal, key=lambda index: stroke_sizes[index][0])
            line_y = float(centers[line_index][1])
            dots_above = any(float(centers[index][1]) < line_y for index in small_strokes)
            dots_below = any(float(centers[index][1]) > line_y for index in small_strokes)
            if dots_above and dots_below:
                return "/"

    if len(valid_strokes) <= 2 and aspect_ratio > 2.2:
        if all(horizontal_lines):
            return "-"

    if len(valid_strokes) == 1 and not (
        is_horizontal(angles[0]) or is_vertical(angles[0])
    ):
        return "/"

    if not 0.7 - 1e-6 <= aspect_ratio <= 1.4 + 1e-6:
        return None

    def centers_overlap(first: int, second: int) -> bool:
        return (
            abs(float(centers[first][0] - centers[second][0])) <= max(width * 0.3, 1e-6)
            and abs(float(centers[first][1] - centers[second][1])) <= max(height * 0.3, 1e-6)
        )

    for first in range(len(valid_strokes)):
        for second in range(first + 1, len(valid_strokes)):
            if not centers_overlap(first, second):
                continue
            first_angle, second_angle = angles[first], angles[second]
            if (is_horizontal(first_angle) and is_vertical(second_angle)) or (
                is_vertical(first_angle) and is_horizontal(second_angle)
            ):
                return "+"

            angle_difference = abs(first_angle - second_angle)
            crossing_angle = min(angle_difference, 180 - angle_difference)
            if (
                not any(
                    is_horizontal(angle) or is_vertical(angle)
                    for angle in (first_angle, second_angle)
                )
                and crossing_angle >= 25
            ):
                return "*"
    return None


def predict_digit(strokes: Stroke) -> list[tuple[int, float]]:
    """Return the three most likely digits as (digit, confidence_percent)."""
    if _digit_model is None:
        raise RuntimeError(
            f"Digit model is unavailable at {_MODEL_PATH}. Run train.py before predicting."
        )

    image = _strokes_to_image(strokes).astype(np.float32) / 255.0
    image = (image - 0.1307) / 0.3081
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(_digit_model(tensor), dim=1)[0]
        confidences, digits = torch.topk(probabilities, k=3)
    return [
        (int(digit), float(confidence) * 100)
        for digit, confidence in zip(digits.tolist(), confidences.tolist())
    ]


def _as_landmarks(landmarks: Iterable[object]) -> list[Landmark]:
    result: list[Landmark] = []
    for point in landmarks:
        if hasattr(point, "x"):
            result.append((float(point.x), float(point.y), float(point.z)))
        else:
            x, y, *rest = point  # type: ignore[misc]
            result.append((float(x), float(y), float(rest[0]) if rest else 0.0))
    if len(result) < 21:
        raise ValueError("A hand pose must contain all 21 MediaPipe landmarks.")
    return result


def _is_thumb_extended(landmarks: Sequence[Landmark]) -> bool:
    tip = landmarks[4]
    ip = landmarks[3]
    index_mcp = landmarks[5]
    tip_distance = ((tip[0] - index_mcp[0]) ** 2 + (tip[1] - index_mcp[1]) ** 2) ** 0.5
    ip_distance = ((ip[0] - index_mcp[0]) ** 2 + (ip[1] - index_mcp[1]) ** 2) ** 0.5
    return tip_distance > ip_distance * 1.15


def _classify_gesture(landmarks: Sequence[Landmark]) -> DrawingMode:
    fingers_extended = [
        landmarks[tip][1] < landmarks[pip][1] - 0.015
        for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)
    ]
    index_extended, middle_extended, ring_extended, pinky_extended = fingers_extended
    thumb_extended = _is_thumb_extended(landmarks)

    if not any(fingers_extended) and not thumb_extended:
        return DrawingMode.CLEAR
    if all(fingers_extended) and thumb_extended:
        return DrawingMode.PAUSED
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return DrawingMode.PAUSED if thumb_extended else DrawingMode.ERASER
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return DrawingMode.PAUSED if thumb_extended else DrawingMode.DRAWING
    return DrawingMode.IDLE


def _distance_to_segment(
    cursor: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared == 0:
        return float(np.hypot(cursor[0] - start[0], cursor[1] - start[1]))
    projection = (
        (cursor[0] - start[0]) * segment_x + (cursor[1] - start[1]) * segment_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = (start[0] + projection * segment_x, start[1] + projection * segment_y)
    return float(np.hypot(cursor[0] - closest[0], cursor[1] - closest[1]))


class HandPoseTracker:
    """Stateful MediaPipe tracker; create one instance per camera or upload session."""

    def __init__(self) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe and OpenCV are required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

        self._cv2 = cv2
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._strokes: Stroke = []
        self._previous_mode = DrawingMode.IDLE
        self._lock = Lock()

    def process_drawing_stroke(
        self,
        landmarks: Iterable[object],
        frame_size: tuple[int, int] = (640, 480),
    ) -> Stroke:
        points = _as_landmarks(landmarks)
        mode = _classify_gesture(points)
        with self._lock:
            if mode == DrawingMode.CLEAR:
                self._strokes.clear()
            elif mode == DrawingMode.DRAWING:
                if self._previous_mode != DrawingMode.DRAWING or not self._strokes:
                    self._strokes.append([])
                self._strokes[-1].append((points[8][0], points[8][1]))
            elif mode == DrawingMode.ERASER:
                width, height = frame_size
                cursors = (
                    (points[8][0] * width, points[8][1] * height),
                    (points[12][0] * width, points[12][1] * height),
                )
                self._strokes = self._erase_near_cursors(
                    self._strokes, cursors, width, height
                )
            self._previous_mode = mode
            return [stroke.copy() for stroke in self._strokes]

    @staticmethod
    def _erase_near_cursors(
        strokes: Stroke,
        cursors: Sequence[tuple[float, float]],
        width: int,
        height: int,
    ) -> Stroke:
        remaining: Stroke = []
        for stroke in strokes:
            current: list[tuple[float, float]] = []
            for point in stroke:
                point_px = (point[0] * width, point[1] * height)
                near_cursor = any(
                    np.hypot(point_px[0] - cursor[0], point_px[1] - cursor[1])
                    <= _ERASER_RADIUS_PX
                    for cursor in cursors
                )
                if near_cursor:
                    if current:
                        remaining.append(current)
                        current = []
                    continue

                if current:
                    start_px = (current[-1][0] * width, current[-1][1] * height)
                    intersects = any(
                        _distance_to_segment(cursor, start_px, point_px)
                        <= _ERASER_RADIUS_PX
                        for cursor in cursors
                    )
                    if intersects:
                        remaining.append(current)
                        current = []
                current.append(point)
            if current:
                remaining.append(current)
        return remaining

    def get_strokes(self) -> Stroke:
        with self._lock:
            return [stroke.copy() for stroke in self._strokes]

    def clear_strokes(self) -> None:
        with self._lock:
            self._strokes.clear()
            self._previous_mode = DrawingMode.IDLE

    def predict_hand_pose(self, frame: object) -> tuple[DrawingMode, list[Landmark], Stroke]:
        with self._lock:
            rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            with self._lock:
                self._previous_mode = DrawingMode.IDLE
                return DrawingMode.IDLE, [], [stroke.copy() for stroke in self._strokes]

        landmarks = _as_landmarks(result.multi_hand_landmarks[0].landmark)
        mode = _classify_gesture(landmarks)
        height, width = frame.shape[:2]
        strokes = self.process_drawing_stroke(landmarks, (width, height))
        return mode, landmarks, strokes

    def close(self) -> None:
        self._hands.close()


_default_tracker: HandPoseTracker | None = None
_default_tracker_lock = Lock()


def _get_default_tracker() -> HandPoseTracker:
    global _default_tracker
    with _default_tracker_lock:
        if _default_tracker is None:
            _default_tracker = HandPoseTracker()
        return _default_tracker


def process_drawing_stroke(landmarks: Iterable[object]) -> Stroke:
    """Update the default tracker's stroke history from 21 hand landmarks."""
    return _get_default_tracker().process_drawing_stroke(landmarks)


def predict_hand_pose(frame: object) -> tuple[DrawingMode, list[Landmark], Stroke]:
    """Detect one hand in a BGR image and return mode, landmarks, and strokes."""
    return _get_default_tracker().predict_hand_pose(frame)
