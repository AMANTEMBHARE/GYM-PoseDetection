import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ---------------- MediaPipe Setup ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- Angle Function ----------------
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


# ---------------- Main Dataset Creator ----------------
def process_video(video_path, exercise_label):
    cap = cv2.VideoCapture(video_path)

    frame_no = 0
    rows = []

    prev_left_knee = None
    prev_right_knee = None
    prev_back = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # LEFT side joints
            l_shoulder, lsh_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            l_hip, lh_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
            l_knee, lk_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
            l_ankle, la_vis = get_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

            # RIGHT side joints
            r_shoulder, rsh_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            r_hip, rh_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
            r_knee, rk_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
            r_ankle, ra_vis = get_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # Check visibility
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

                vertical_point = [mid_hip[0], max(mid_hip[1] - 0.2, 0)]
                back_angle = calculate_angle(mid_shoulder, mid_hip, vertical_point)

                # Symmetry difference
                knee_diff = abs(left_knee_angle - right_knee_angle)
                hip_diff = abs(left_hip_angle - right_hip_angle)

                # Velocity features
                left_knee_vel = 0 if prev_left_knee is None else left_knee_angle - prev_left_knee
                right_knee_vel = 0 if prev_right_knee is None else right_knee_angle - prev_right_knee
                back_vel = 0 if prev_back is None else back_angle - prev_back

                prev_left_knee = left_knee_angle
                prev_right_knee = right_knee_angle
                prev_back = back_angle

                rows.append([
                    os.path.basename(video_path),
                    frame_no,
                    left_knee_angle,
                    right_knee_angle,
                    left_hip_angle,
                    right_hip_angle,
                    back_angle,
                    knee_diff,
                    hip_diff,
                    left_knee_vel,
                    right_knee_vel,
                    back_vel,
                    exercise_label
                ])

    cap.release()
    return rows


def create_dataset(dataset_folder, output_csv="gym_pose_dataset.csv"):
    all_rows = []

    for exercise_name in os.listdir(dataset_folder):
        exercise_path = os.path.join(dataset_folder, exercise_name)

        if not os.path.isdir(exercise_path):
            continue

        print(f"\nProcessing Exercise: {exercise_name}")

        for video_file in os.listdir(exercise_path):
            if video_file.endswith(".mp4") or video_file.endswith(".avi") or video_file.endswith(".mov"):
                video_path = os.path.join(exercise_path, video_file)

                print(f"  -> Processing video: {video_file}")
                rows = process_video(video_path, exercise_name)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "video_name",
        "frame",
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
        "back_angle",
        "knee_diff",
        "hip_diff",
        "left_knee_vel",
        "right_knee_vel",
        "back_vel",
        "exercise"
    ])

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved as: {output_csv}")


# ---------------- Run ----------------
if __name__ == "__main__":
    dataset_folder = "dataset_videos"   # change if needed
    create_dataset(dataset_folder, "gym_pose_dataset.csv")
88