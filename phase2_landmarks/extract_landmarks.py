import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def get_landmark_xy(landmarks, landmark_id):
    lm = landmarks[landmark_id]
    return lm.x, lm.y, lm.visibility


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key joints
        joints = {
            "L_SHOULDER": get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            "R_SHOULDER": get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),

            "L_HIP": get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value),
            "R_HIP": get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value),

            "L_KNEE": get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value),
            "R_KNEE": get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value),

            "L_ANKLE": get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value),
            "R_ANKLE": get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        }

        # Draw skeleton
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display landmark points
        for name, (x, y, vis) in joints.items():
            px, py = int(x * w), int(y * h)

            if vis > 0.5:
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(frame, name, (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Phase 2 - Landmark Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
