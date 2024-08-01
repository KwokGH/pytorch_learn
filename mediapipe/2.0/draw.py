
import mediapipe as mp

import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import matplotlib.pyplot as plt
import math
import os
import enum

def draw_landmarks_on_image(rgb_image, detection_result,frame_score_val,is_koufen):
    annotated_image = np.copy(rgb_image)

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in detection_result
    ])

    image_height, image_width, _ = annotated_image.shape
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils

        # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    thickness = 2
    line_type = cv2.LINE_AA
    line_spacing = 10

    # 获取默认的关键点样式
    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    
    for item in frame_score_val:
        score_points = item["point"]
        if len(score_points)<3:
            continue
        a = score_points[0]
        b = score_points[1]
        c = score_points[2]
        elbow2 = [detection_result[b].x * image_width,
        detection_result[b].y * image_height]
        cv2.putText(annotated_image, str(int(100)),
                            tuple(np.multiply(elbow2, [1, 1]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                            )
    if len(frame_score_val)>0:
        first_item = frame_score_val[0]
        texts = [
        f"leibie:{first_item['类别']}",
        f"label:{first_item['video_label']}",
        f"koufen:{is_koufen}"]
        # 初始文本位置（右上角）
        text_x = image_width - 10
        text_y = 30

        # 在帧上逐行绘制文本
        for i, text in enumerate(texts):
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_y_pos = text_y + i * (text_size[1] + line_spacing)
            cv2.putText(annotated_image, text, (text_x - text_size[0], text_y_pos), font, font_scale, font_color, thickness, line_type)


        # 修改所有关键点的颜色
    point_color = (0, 0, 255)
    line_color = (0, 255, 255)  # 你想要的颜色 (BGR格式)
    line_thickness = 1  # 设置线条厚度
    for landmark in pose_landmark_style:
        pose_landmark_style[landmark] = mp_drawing.DrawingSpec(color=point_color, thickness=1, circle_radius=4)
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        pose_landmark_style,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=line_thickness))
    return annotated_image
