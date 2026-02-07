import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,      # video stream
    model_complexity=1,           # balanced speed & accuracy
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils
