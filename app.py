"""Streamlit interface for air-writing hand tracking."""

from __future__ import annotations

import ast
from threading import Lock

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from classifier import (
    DrawingMode,
    HandPoseTracker,
    classify_operator,
    is_open_palm,
    is_pinky_only,
    predict_digit,
)

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _evaluate_expression(expression: str) -> int | float:
    """Evaluate arithmetic expressions using only numeric constants and basic operators."""
    expression = expression.replace("×", "*").replace("÷", "/")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("Unsupported arithmetic expression")

    return evaluate(tree.body)


def _calculate_retained_expression() -> None:
    expression = "".join(map(str, st.session_state.recognized_digits))
    if expression.endswith("="):
        expression = expression[:-1]
    try:
        st.session_state["calculation_result"] = _evaluate_expression(expression)
        st.session_state["calculation_error"] = False
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        st.session_state["calculation_result"] = None
        st.session_state["calculation_error"] = True


def _append_retained_symbol(symbol: str) -> None:
    st.session_state.recognized_digits.append(symbol)
    st.session_state.recognized_confidences.append(None)
    st.session_state["calculation_result"] = None
    st.session_state["calculation_error"] = False
    if symbol == "=":
        _calculate_retained_expression()


def _annotate_frame(
    frame: np.ndarray,
    mode: DrawingMode,
    landmarks: list[tuple[float, float, float]],
    strokes: list[list[tuple[float, float]]],
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    for stroke in strokes:
        for start, end in zip(stroke, stroke[1:]):
            start_px = (int(start[0] * width), int(start[1] * height))
            end_px = (int(end[0] * width), int(end[1] * height))
            cv2.line(output, start_px, end_px, (0, 220, 255), 4, cv2.LINE_AA)

    if landmarks:
        for start, end in _HAND_CONNECTIONS:
            start_px = (int(landmarks[start][0] * width), int(landmarks[start][1] * height))
            end_px = (int(landmarks[end][0] * width), int(landmarks[end][1] * height))
            cv2.line(output, start_px, end_px, (80, 200, 80), 2, cv2.LINE_AA)
        for x, y, _ in landmarks:
            cv2.circle(output, (int(x * width), int(y * height)), 3, (255, 120, 60), -1)
        cursor_tips = (8, 12) if mode == DrawingMode.ERASER else (8,)
        for tip in cursor_tips:
            tip_x, tip_y, _ = landmarks[tip]
            cursor = (int(tip_x * width), int(tip_y * height))
            if mode == DrawingMode.ERASER:
                cv2.circle(output, cursor, 18, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(output, cursor, 8, (0, 0, 255), -1)

    cv2.putText(
        output,
        f"Mode: {mode.value}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


class AirWritingVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.tracker = HandPoseTracker()
        self._gesture_lock = Lock()
        self._palm_detected = False
        self._pinky_detected = False
        self._retain_requested = False
        self._undo_requested = False

    def consume_gesture_requests(self) -> tuple[bool, bool]:
        with self._gesture_lock:
            requests = self._retain_requested, self._undo_requested
            self._retain_requested = False
            self._undo_requested = False
            return requests

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        mode, landmarks, strokes = self.tracker.predict_hand_pose(image)
        palm_detected = bool(landmarks) and is_open_palm(landmarks)
        pinky_detected = bool(landmarks) and is_pinky_only(landmarks)
        with self._gesture_lock:
            if palm_detected and not self._palm_detected:
                self._retain_requested = True
            if pinky_detected and not self._pinky_detected:
                self._undo_requested = True
            self._palm_detected = palm_detected
            self._pinky_detected = pinky_detected
        annotated = _annotate_frame(image, mode, landmarks, strokes)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def _retain_current_drawing(
    video_processor: AirWritingVideoProcessor | None,
) -> None:
    strokes = video_processor.tracker.get_strokes() if video_processor else []
    predictions = st.session_state.get("digit-predictions")
    retained = False
    if strokes:
        operator = classify_operator(strokes)
        if operator is not None:
            _append_retained_symbol(operator)
            st.session_state.pop("digit-predictions", None)
            st.session_state.pop("operator-prediction", None)
            retained = True
            if video_processor is not None:
                video_processor.tracker.clear_strokes()
        else:
            st.session_state.pop("operator-prediction", None)
            try:
                predictions = predict_digit(strokes)
            except (RuntimeError, ValueError) as error:
                st.error(str(error))
                predictions = None
    if not retained and predictions:
        digit, confidence = predictions[0]
        st.session_state.recognized_digits.append(digit)
        st.session_state.recognized_confidences.append(confidence)
        st.session_state["calculation_result"] = None
        st.session_state["calculation_error"] = False
        retained = True
        if video_processor is not None:
            video_processor.tracker.clear_strokes()
        st.session_state.pop("digit-predictions", None)
    if not retained:
        st.warning("先に数字または記号を描いて判定してください。")


def _undo_last_character(
    video_processor: AirWritingVideoProcessor | None,
) -> None:
    if st.session_state.recognized_digits:
        st.session_state.recognized_digits.pop()
    if st.session_state.recognized_confidences:
        st.session_state.recognized_confidences.pop()
    st.session_state["calculation_result"] = None
    st.session_state["calculation_error"] = False
    st.session_state.pop("digit-predictions", None)
    st.session_state.pop("operator-prediction", None)
    if video_processor is not None:
        video_processor.tracker.clear_strokes()


st.set_page_config(page_title="空中手書き電卓", page_icon="✍️", layout="wide")
st.title("空中手書き電卓")
st.caption("CNN × 幾何ルールによるシームレス入力")
st.write(
    "人差し指だけで描画、人差し指と中指で消去します。"
    "親指＋人差し指、親指＋人差し指＋中指で一時停止し、パーで判定・保持、"
    "グーで描画を消去、小指だけを立てると1文字消去します。"
)
st.session_state.setdefault("recognized_digits", [])
st.session_state.setdefault("recognized_confidences", [])
st.session_state.setdefault("calculation_result", None)
st.session_state.setdefault("calculation_error", False)

st.caption("ブラウザーからWebカメラへのアクセスを許可してください。")
webrtc_context = webrtc_streamer(
    key="air-writing-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=AirWritingVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


@st.fragment(run_every="200ms")
def _watch_for_gestures() -> None:
    video_processor = webrtc_context.video_processor
    if video_processor is not None:
        retain_requested, undo_requested = video_processor.consume_gesture_requests()
        if undo_requested:
            _undo_last_character(video_processor)
            st.rerun(scope="app")
        if retain_requested:
            _retain_current_drawing(video_processor)
            st.rerun(scope="app")


_watch_for_gestures()

predict_col, retain_col = st.columns(2)
with predict_col:
    if st.button("判定する", key="predict-digit"):
        st.session_state.pop("digit-predictions", None)
        st.session_state.pop("operator-prediction", None)
        video_processor = webrtc_context.video_processor
        if video_processor is None:
            st.warning("先にWebカメラを開始して数字を描いてください。")
        else:
            strokes = video_processor.tracker.get_strokes()
            operator = classify_operator(strokes)
            if operator is not None:
                st.session_state["operator-prediction"] = operator
            else:
                st.session_state.pop("operator-prediction", None)
                try:
                    st.session_state["digit-predictions"] = predict_digit(strokes)
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))
with retain_col:
    if st.button("判定・保持", key="retain-digit"):
        _retain_current_drawing(webrtc_context.video_processor)
operator_columns = st.columns(5)
for column, (label, symbol) in zip(
    operator_columns,
    (("＋", "+"), ("－", "-"), ("×", "*"), ("÷", "/"), ("＝", "=")),
):
    with column:
        if st.button(label, key=f"append-symbol-{symbol}"):
            _append_retained_symbol(symbol)
if st.button("計算実行（＝）", key="calculate-expression"):
    _calculate_retained_expression()
reset_col, undo_col = st.columns(2)
with reset_col:
    if st.button("全リセット", key="reset-all"):
        video_processor = webrtc_context.video_processor
        if video_processor is not None:
            video_processor.tracker.clear_strokes()
        st.session_state.recognized_digits.clear()
        st.session_state.recognized_confidences.clear()
        st.session_state["calculation_result"] = None
        st.session_state["calculation_error"] = False
        st.session_state.pop("digit-predictions", None)
        st.session_state.pop("operator-prediction", None)
with undo_col:
    if st.button("1文字消去（取り消し）", key="undo-last-character"):
        _undo_last_character(webrtc_context.video_processor)

predictions = st.session_state.get("digit-predictions")
operator_prediction = st.session_state.get("operator-prediction")
if operator_prediction:
    st.subheader("判定結果")
    st.metric("記号", operator_prediction)
elif predictions:
    st.subheader("判定結果")
    metric_columns = st.columns(3)
    for column, (digit, confidence) in zip(metric_columns, predictions):
        column.metric(f"数字 {digit}", f"{confidence:.1f}%")

st.markdown(f"## 入力結果: {''.join(map(str, st.session_state.recognized_digits))}")
if st.session_state["calculation_error"]:
    st.warning("計算エラー")
elif st.session_state["calculation_result"] is not None:
    st.markdown(f"## **計算結果: {st.session_state['calculation_result']}**")

if st.session_state.recognized_digits:
    st.subheader("保持した数字・記号の履歴")
    for index, (digit, confidence) in enumerate(
        zip(st.session_state.recognized_digits, st.session_state.recognized_confidences),
        start=1,
    ):
        confidence_label = f" [{confidence:.1f}%]" if confidence is not None else ""
        st.write(f"{index}文字目: {digit}{confidence_label}")
