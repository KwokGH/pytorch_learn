import mediapipe as mp

import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import matplotlib.pyplot as plt
import math
import os
import enum
import time

class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32


# 初始化模型
def initModel(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    
    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker

def calculate_angle(a, b, c):
    a = np.array(a)  # 第一个点
    b = np.array(b)  # 第二个点（角的顶点）
    c = np.array(c)  # 第三个点

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle
# 绘制线条
def draw_landmarks_on_image(rgb_image, detection_result,keyspointsAngle):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    # 获取默认的关键点样式
    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()

    # 修改样式（例如，更改关键点的颜色）
    # pose_landmark_style[mp_pose.PoseLandmark.LEFT_SHOULDER] = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)
    # pose_landmark_style[mp_pose.PoseLandmark.RIGHT_SHOULDER] = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

    # 修改所有关键点的颜色
    point_color = (0, 0, 255)
    line_color = (0, 255, 255)  # 你想要的颜色 (BGR格式)
    line_thickness = 1  # 设置线条厚度
    for landmark in pose_landmark_style:
        pose_landmark_style[landmark] = mp_drawing.DrawingSpec(color=point_color, thickness=1, circle_radius=4)

    image_height, image_width, _ = annotated_image.shape
    
    for keys in keyspointsAngle:
      a = keys[0]
      b = keys[1]
      c = keys[2]
      shoulder = [pose_landmarks[a].x * image_width,
                pose_landmarks[a].y * image_height]
      elbow = [pose_landmarks[b].x * image_width,
              pose_landmarks[b].y * image_height]
      wrist = [pose_landmarks[c].x * image_width,
              pose_landmarks[c].y * image_height]
      angle = calculate_angle(shoulder, elbow, wrist)
      cv2.putText(annotated_image, str(int(angle)),
                        tuple(np.multiply(elbow, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      pose_landmark_style,
      connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=line_thickness))
  return annotated_image


def process_angle_rule(rule_item,p1,p2,p3):
  score = 0
  max = rule_item['max']
  min = rule_item['min']
  get = calculate_angle(p1, p2, p3)

  score = proximity_to_interval(get,min,max)
  # if get>=min and get<=max:
  #     score = 1
  # else:
  #     score = 0

  return score,get

def process_height_rule(rule_item,p1,p2,p3):
  score = 0
  max = rule_item['max']
  min = rule_item['min']
  y1 = p1[1]
  y2 = p2[1]
  get = y1-y2

  score = proximity_to_interval(get,min,max)
  # if get>=min and get<=max:
  #     score = 1
  # else:
  #     score = 0

  return score,get

def proximity_to_interval(value, min_val, max_val):
    if min_val > max_val:
        raise ValueError("min_val should not be greater than max_val")

    if min_val <= value <= max_val:
        return 1.0  # 值在区间内，返回1

    # 计算值到区间最近边界的距离
    distance_to_closest_bound = min(abs(value - min_val), abs(value - max_val))
    interval_range = max_val - min_val
    
    # 归一化距离，使其返回值接近1
    normalized_distance = distance_to_closest_bound / interval_range
    proximity = 1 / (1 + normalized_distance)  # 将距离映射到接近1的值

    return proximity

def process_weight_rule(rule_item,p1,p2,p3):
  score = 0
  max = rule_item['max']
  min = rule_item['min']
  x1 = p1[0]
  x2 = p2[0]
  get = x1-x2

  score = proximity_to_interval(get,min,max)
  # if get>=min and get<=max:
  #     score = 1
  # else:
  #     score = 0

  return score,get

rule_func = {
  #  'limit_angle':process_angle_rule,
  #  "limit_height":process_height_rule,
   "angle":process_angle_rule,
   "height":process_height_rule,
   "weight":process_weight_rule
}

def get_video_landmarker(numpy_image,landmarker):
   # 姿势特征点检测
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    detection_result = landmarker.detect(mp_image)    
    # detection_result.pose_world_landmarks    
    return detection_result.pose_landmarks[0]

def process_rules(numpy_image,rule,detection_result):
    # 姿势特征点检测
    pose_landmarks = detection_result
    image_height, image_width, _ = numpy_image.shape

    total_score = dict()
    total_score["limit_angle"]=[]
    total_score["limit_height"]=[]
    total_score["angle"]=[]
    total_score["height"]=[]

    keyspointsAngle=[]

    for rule_item in rule['rule']:
      keys = rule_item['point']
      p1,p2,p3 = 0,0,0
      a,b,c = 0,0,0

      if len(keys)==2:
        a = keys[0]
        b = keys[1]        
      elif len(keys)==3:
        a = keys[0]
        b = keys[1]
        c = keys[2]
        keyspointsAngle.append(keys)                     

      p1 = [pose_landmarks[a].x * image_width,
                pose_landmarks[a].y * image_height]
      p2 = [pose_landmarks[b].x * image_width,
              pose_landmarks[b].y * image_height]
      p3 = [pose_landmarks[c].x * image_width,
              pose_landmarks[c].y * image_height]
      
      score,val = rule_func[rule_item['type']](rule_item,p1,p2,p3)
   
      if rule_item['type'] not in total_score:
         total_score[rule_item['type']] = []
      total_score[rule_item['type']].append({
         "score":score,
         "val":val,
        #  "frame":numpy_image
      })
      # if score==1:
      #   cv2.imshow('Image', numpy_image)
      #   # 等待键盘事件，参数为等待时间（毫秒）
      #   cv2.waitKey(0)  # 等待直到有键盘事件
      #   cv2.destroyAllWindows()

        
        # print(get_angle,want_angle,deviation,want_angle-get_angle)
        # if abs(want_angle-get_angle)<=deviation:
        #     cv2.imshow('Image', numpy_image)
        #     # 等待键盘事件，参数为等待时间（毫秒）
        #     cv2.waitKey(0)  # 等待直到有键盘事件
        #     cv2.destroyAllWindows()
    
    annotated_image = draw_landmarks_on_image(numpy_image,detection_result,keyspointsAngle)
    
    
    return {
       "data":total_score,
       "annotated_image":annotated_image
    }

