import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh


def get_face_angle(landmarks, img_w=1920, img_h=1080):
    face_3d = []
    face_2d = []
    for idx, lm in enumerate(landmarks):
        if idx in [1, 33, 61, 199, 263, 291]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    # NumPy配列に変換する
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )
    # 歪みパラメータ
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rotation_vecor, translation_vector = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )
    # 回転行列を取得する
    rotation_mat, _ = cv2.Rodrigues(rotation_vecor)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_mat)
    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # 回転行列からオイラー角に変換
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return roll, pitch, yaw


cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 画像を水平方向に反転し、BGRからRGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 顔の角度を計算
                roll, pitch, yaw = get_face_angle(face_landmarks.landmark)
                print()
                # 正面判定
                if abs(roll) + abs(pitch) + abs(yaw) <= 0.0099:
                    text = "OK"
                    color = (0, 255, 0)
                else:
                    text = "No"
                    color = (0, 0, 255)

                # 正面判定結果を表示
                cv2.putText(
                    image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        cv2.imshow("MediaPipe Face Mesh", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
