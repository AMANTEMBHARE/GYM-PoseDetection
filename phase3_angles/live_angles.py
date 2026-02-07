import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def get_point(landmarks, idx):
    lm = landmarks[idx]
    return [lm.x, lm.y], lm.visibility


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

        # LEFT side
        l_shoulder, lsh_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        l_hip, lh_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
        l_knee, lk_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
        l_ankle, la_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

        # RIGHT side
        r_shoulder, rsh_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        r_hip, rh_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
        r_knee, rk_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
        r_ankle, ra_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        if (lsh_vis > 0.5 and lh_vis > 0.5 and lk_vis > 0.5 and la_vis > 0.5 and
            rsh_vis > 0.5 and rh_vis > 0.5 and rk_vis > 0.5 and ra_vis > 0.5):

            # Knee angles
            left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Hip angles
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

            # Mid points for back angle
            mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2,
                            (l_shoulder[1] + r_shoulder[1]) / 2]

            mid_hip = [(l_hip[0] + r_hip[0]) / 2,
                       (l_hip[1] + r_hip[1]) / 2]

            vertical_point = [mid_hip[0], mid_hip[1] - 0.2]
            back_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)

            # Display angles
            cv2.putText(frame, f"Left Knee: {int(left_knee_angle)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Right Knee: {int(right_knee_angle)}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Left Hip: {int(left_hip_angle)}",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Right Hip: {int(right_hip_angle)}",
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Back Angle: {int(back_angle)}",
                        (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Symmetry difference
            knee_diff = abs(left_knee_angle - right_knee_angle)
            hip_diff = abs(left_hip_angle - right_hip_angle)

            cv2.putText(frame, f"Knee Diff: {int(knee_diff)}",
                        (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Hip Diff: {int(hip_diff)}",
                        (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Phase 3 - Full Angle Calculation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
