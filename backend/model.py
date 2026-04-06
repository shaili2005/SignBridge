from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp


@dataclass
class PredictionResult:
    gesture: str
    text: str
    confidence: float
    explanation: str
    modalities: dict[str, Any]


class SignRecognizer:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.face = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def predict(self, frame) -> PredictionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = self.hands.process(rgb)
        face_result = self.face.process(rgb)
        pose_result = self.pose.process(rgb)

        hand_features = self._extract_hand_features(hand_result)
        face_features = self._extract_face_features(face_result)
        pose_features = self._extract_pose_features(pose_result)

        gesture, confidence, explanation_parts = self._classify(
            hand_features,
            face_features,
            pose_features,
        )

        return PredictionResult(
            gesture=gesture,
            text=self._gesture_to_text(gesture),
            confidence=round(confidence, 2),
            explanation=" | ".join(explanation_parts),
            modalities={
                "hand": hand_features,
                "face": face_features,
                "pose": pose_features,
            },
        )

    def _extract_hand_features(self, hand_result) -> dict[str, Any]:
        default = {
            "detected": False,
            "finger_states": {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            },
            "open_palm": False,
            "closed_fist": False,
            "pinch": False,
            "ok_sign": False,
            "thumbs_up": False,
            "thumbs_down": False,
            "pointing_up": False,
            "peace": False,
            "i_love_you": False,
            "wrist_y": None,
            "thumb_down_candidate": False,
        }

        if not hand_result.multi_hand_landmarks:
            return default

        hand_landmarks = hand_result.multi_hand_landmarks[0].landmark
        wrist = hand_landmarks[0]
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        thumb_mcp = hand_landmarks[2]
        index_tip = hand_landmarks[8]
        index_dip = hand_landmarks[7]
        index_pip = hand_landmarks[6]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]
        index_base = hand_landmarks[5]
        middle_base = hand_landmarks[9]
        ring_base = hand_landmarks[13]
        pinky_base = hand_landmarks[17]

        ref_dist = abs(wrist.y - middle_base.y) + 1e-6
        thumb_vertical = (thumb_mcp.y - thumb_tip.y) / ref_dist
        thumb_horizontal = abs(thumb_tip.x - thumb_ip.x)
        index_vertical = (index_base.y - index_tip.y) / ref_dist
        thumb_extension = self._distance(thumb_tip, thumb_mcp) / ref_dist
        index_extended = (
            index_vertical > 0.45
            and index_tip.y < index_dip.y
            and index_dip.y < index_pip.y
        )
        thumb_down_candidate = (
            thumb_tip.y > wrist.y + (0.08 * ref_dist)
            and thumb_tip.y > thumb_mcp.y + (0.05 * ref_dist)
            and thumb_tip.y > max(index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y) + (0.03 * ref_dist)
            and thumb_extension > 0.75
            and middle_tip.y > middle_base.y - (0.05 * ref_dist)
            and ring_tip.y > ring_base.y - (0.05 * ref_dist)
            and pinky_tip.y > pinky_base.y - (0.05 * ref_dist)
        )
        finger_states = {
            "thumb": thumb_horizontal > 0.045 or thumb_vertical > 0.55,
            "index": index_extended,
            "middle": (middle_base.y - middle_tip.y) / ref_dist > 0.4,
            "ring": (ring_base.y - ring_tip.y) / ref_dist > 0.35,
            "pinky": (pinky_base.y - pinky_tip.y) / ref_dist > 0.3,
        }

        pinch = (
            self._distance(thumb_tip, index_tip) < 0.12
            and self._distance(thumb_tip, index_dip) < 0.18
            and index_tip.y > index_base.y - (0.1 * ref_dist)
            and finger_states["middle"]
            and finger_states["ring"]
            and finger_states["pinky"]
        )
        ok_sign = (
            (
                self._distance(thumb_tip, index_tip) < 0.16
                or self._distance(thumb_tip, index_dip) < 0.2
            )
            and finger_states["middle"]
            and finger_states["ring"]
            and finger_states["pinky"]
        )
        open_palm = all(
            finger_states[finger]
            for finger in ("index", "middle", "ring", "pinky")
        )
        closed_fist = not any(
            finger_states[finger]
            for finger in ("index", "middle", "ring", "pinky")
        ) and not finger_states["thumb"]
        thumbs_up = (
            finger_states["thumb"]
            and not finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
            and thumb_vertical > 0.55
            and thumb_tip.y < wrist.y
        )
        thumbs_down = (
            thumb_down_candidate
            and (thumb_tip.y - thumb_mcp.y) / ref_dist > 0.1
        )
        pointing_up = (
            finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
            and index_tip.y < wrist.y
            and not thumb_down_candidate
        )
        peace = (
            finger_states["index"]
            and finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        )
        i_love_you = (
            finger_states["thumb"]
            and finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and finger_states["pinky"]
        )

        return {
            "detected": True,
            "finger_states": finger_states,
            "open_palm": open_palm,
            "closed_fist": closed_fist,
            "pinch": pinch,
            "ok_sign": ok_sign,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "pointing_up": pointing_up,
            "peace": peace,
            "i_love_you": i_love_you,
            "wrist_y": wrist.y,
            "thumb_down_candidate": thumb_down_candidate,
        }

    def _extract_face_features(self, face_result) -> dict[str, Any]:
        default = {
            "detected": False,
            "centered": False,
            "score": 0.0,
        }

        if not face_result.detections:
            return default

        detection = face_result.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_center_x = bbox.xmin + (bbox.width / 2)
        face_center_y = bbox.ymin + (bbox.height / 2)
        centered = 0.3 <= face_center_x <= 0.7 and 0.2 <= face_center_y <= 0.7
        score = detection.score[0] if detection.score else 0.0

        return {
            "detected": True,
            "centered": centered,
            "score": round(score, 2),
        }

    def _extract_pose_features(self, pose_result) -> dict[str, Any]:
        default = {
            "detected": False,
            "shoulders_visible": False,
            "upright": False,
            "hand_above_shoulder": False,
            "shoulder_mid_y": None,
        }

        if not pose_result.pose_landmarks:
            return default

        landmarks = pose_result.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]

        shoulders_visible = (
            left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5
        )
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        upright = shoulders_visible and abs(left_shoulder.y - right_shoulder.y) < 0.08
        hand_above_shoulder = (
            left_wrist.visibility > 0.4 and left_wrist.y < shoulder_mid_y
        ) or (
            right_wrist.visibility > 0.4 and right_wrist.y < shoulder_mid_y
        )

        return {
            "detected": True,
            "shoulders_visible": shoulders_visible,
            "upright": upright and nose.visibility > 0.5,
            "hand_above_shoulder": hand_above_shoulder,
            "shoulder_mid_y": shoulder_mid_y,
        }

    def _classify(
        self,
        hand_features: dict[str, Any],
        face_features: dict[str, Any],
        pose_features: dict[str, Any],
    ) -> tuple[str, float, list[str]]:
        explanation_parts: list[str] = []

        if not hand_features["detected"]:
            return "NO_SIGN", 0.0, ["Hand not clearly detected."]

        fingers = hand_features["finger_states"]
        raised = [name for name, is_up in fingers.items() if is_up]
        explanation_parts.append(
            f"Hand cues: raised fingers -> {', '.join(raised) if raised else 'none'}."
        )

        if face_features["detected"]:
            face_note = "face centered" if face_features["centered"] else "face detected off-center"
            explanation_parts.append(
                f"Face cues: {face_note} (score {face_features['score']})."
            )
        else:
            explanation_parts.append("Face cues: no reliable face detected.")

        if pose_features["detected"]:
            posture_bits = []
            if pose_features["upright"]:
                posture_bits.append("upright posture")
            if pose_features["hand_above_shoulder"]:
                posture_bits.append("raised arm")
            explanation_parts.append(
                "Body cues: "
                + (", ".join(posture_bits) if posture_bits else "neutral upper-body posture")
                + "."
            )
        else:
            explanation_parts.append("Body cues: no reliable upper-body landmarks detected.")

        if (
            hand_features["open_palm"]
            and pose_features["detected"]
            and not pose_features["hand_above_shoulder"]
            and pose_features["shoulder_mid_y"] is not None
            and hand_features["wrist_y"] is not None
            and hand_features["wrist_y"] > pose_features["shoulder_mid_y"] - 0.08
        ):
            explanation_parts.append(
                "Fusion: open palm held below the shoulder line matches PLEASE."
            )
            return "PLEASE", 0.82, explanation_parts

        if hand_features["open_palm"]:
            confidence = 0.88
            if pose_features["hand_above_shoulder"]:
                confidence = 0.93
                explanation_parts.append("Fusion: open palm with raised arm strongly matches HELLO.")
            else:
                explanation_parts.append("Fusion: open palm pattern matches HELLO.")
            return "HELLO", confidence, explanation_parts

        if hand_features["thumbs_down"]:
            explanation_parts.append("Fusion: thumb is clearly directed downward below the other fingertips.")
            return "THUMBS_DOWN", 0.9, explanation_parts

        if hand_features["ok_sign"] or hand_features["pinch"]:
            explanation_parts.append("Fusion: thumb and index form a circle while the other three fingers stay extended.")
            return "OK", 0.87, explanation_parts

        if hand_features["peace"]:
            explanation_parts.append("Fusion: index and middle fingers raised match PEACE.")
            return "PEACE", 0.85, explanation_parts

        if hand_features["i_love_you"]:
            explanation_parts.append(
                "Fusion: thumb, index, and pinky raised together match I LOVE YOU."
            )
            return "I_LOVE_YOU", 0.88, explanation_parts

        if hand_features["thumbs_up"]:
            explanation_parts.append("Fusion: only thumb is extended upward.")
            return "THUMBS_UP", 0.89, explanation_parts

        if hand_features["pointing_up"]:
            explanation_parts.append("Fusion: index finger raised while others are folded.")
            return "YES", 0.86, explanation_parts

        if hand_features["closed_fist"]:
            explanation_parts.append("Fusion: closed fist pattern matches NO.")
            return "NO", 0.84, explanation_parts

        explanation_parts.append("Fusion: gesture pattern is outside the current sign vocabulary.")
        return "UNSURE", 0.35, explanation_parts

    def _gesture_to_text(self, gesture: str) -> str:
        phrases = {
            "HELLO": "Hello",
            "YES": "Yes",
            "NO": "No",
            "OK": "Okay",
            "PEACE": "Peace",
            "THUMBS_UP": "Thumbs up",
            "THUMBS_DOWN": "Thumbs down",
            "PLEASE": "Please",
            "I_LOVE_YOU": "I love you",
            "NO_SIGN": "No sign detected",
            "UNSURE": "Gesture not recognized",
        }
        return phrases.get(gesture, gesture.replace("_", " ").title())

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5
