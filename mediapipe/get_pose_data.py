import argparse
import json
import mediapipe_model
import os
import cv2
import copy
import concurrent.futures
import pose_data
from typing import List
import json

def getData(model_path,video_folder):
      # 加载模型
    landmarker = mediapipe_model.initModel(model_path)   
    data_list = []

    # 遍历读取视频，解析出视频中每一帧的坐标
    # 遍历视频文件 
    listdir = os.listdir(video_folder)

    objects = []
    for file_name in listdir:
        # 如果视频没配置规则，则不对视频处理
        # file_name_no_ext,_ = os.path.splitext(os.path.basename(file_name))
        # if file_name_no_ext not in rule_video_map:
        #     continue
        
        action = process_video(data_list, video_folder,file_name,landmarker)
        objects.append(action)
    
    
    # 将对象数组转换为JSON格式并写入文件
    with open('./2.0/pose_data.json', 'w') as f:
        json.dump(objects, f, cls=CustomEncoder, indent=4)
        
    
    return data_list

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pose_data.Action):
            return obj.to_dict()
        if isinstance(obj, pose_data.Keypoint):
            return {
                'x': obj.x,
                'y': obj.y,
                'z': obj.z,
                'visibility': obj.visibility,
                'presence': obj.presence
            }
        return super().default(obj)

def process_video(data_list,video_folder,file_name,landmarker): 
    file_name_no_ext,_ = os.path.splitext(os.path.basename(file_name))   
    video_path = os.path.join(video_folder, file_name)
    if os.path.exists(video_path)==False:
        print(f"video not found: {video_path}")
        return

    print(f"Processing video : {video_path}")
    i = -1
    id = file_name_no_ext
    label = file_name
    keypoints:List[List[pose_data.Keypoint]]=[]

    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    while True:
        ret, frame = cap.read()
        i = i + 1  
        # 检查是否成功读取到帧
        if not ret:
            break           

        # 在这里处理帧
        detection_result = mediapipe_model.get_video_landmarker(frame,landmarker)
        if len(detection_result)<=0:
            print("没有结果",i)
            continue
        
        point_item: List[pose_data.Keypoint]=[]
        for ret in detection_result:  
            point_item.append(pose_data.Keypoint(ret.x,ret.y,ret.z,ret.visibility,ret.presence))
        
        keypoints.append(point_item)        

    return pose_data.Action(id,label,0,0,keypoints)

    
    # print(action.id,action.label,action.keypoints)


def main():
    model_path="D:\\project\\machineL\\data\\mediapipe\\pose_landmarker_heavy.task"
    video_path = "D:\project\machineL\data\mediapipe\shaonian\\temp\\m"
    getData(model_path,video_path)

if __name__ == "__main__":
	main()
