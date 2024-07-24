import argparse
import json
import mediapipe_model
import os
import cv2
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-c","--config",required=True,help="config file")
args = vars(ap.parse_args())



def main():
    config_path = args["config"]
    config = {}
    # 读取配置文件
    with open(config_path, 'r',encoding='utf-8') as file:
        config = json.load(file)
        
    name = config["name"]
    root_path = config["root_path"]
    model_path = os.path.join(root_path, config["model_path"])
    video_folder = "D:\project\machineL\data\mediapipe\shaonian"#os.path.join(root_path, "")
    data = getData(model_path,video_folder)

    # 提取规则
    rules = list(config["rules"])
    result = process_data_list(data,rules)
    #             "frame_rule_score":frame_rule_score,
    #             "frame_index":frame_index,
    #             "frame":frame,
    #             "pose_landmarks":pose_landmarks

    best = dict()
    for file_id,video_item in result.items():
        best[file_id]=dict()

        frame_temp = dict()
        for frameItem in video_item:                        
            for index,scores in enumerate(frameItem['frame_rule_score']):                
                frame_temp[index] = -1
                temp_score = 0                       
                for scoreItem in scores:                    
                    # temp_score.append((scoreItem['val'],scoreItem['score']))
                    temp_score = temp_score + scoreItem['score']

                if frame_temp[index]<temp_score:
                    print("temp_score",temp_score)
                    frame_temp[index] = temp_score
                    best[file_id][index] = frameItem
                # if best[file_id][frameItem['frame_index']][index]<temp_score:
                #     best[file_id][frameItem['frame_index']][index] = temp_score


                # print("index:",frameItem['frame_index'],"score:",temp_score)
    for idx,fileItem in best.items():        
        for idx2,frameItem in fileItem.items():
            print(frameItem['frame_rule_score'])
            # if isinstance(frameItem, str):
            #     print(frameItem)
            # for item in frameItem.items():
            #     print(item)



def getData(model_path,video_folder):
      # 加载模型
    landmarker = mediapipe_model.initModel(model_path)   
    data_list = []

    # 遍历读取视频，解析出视频中每一帧的坐标
    # 遍历视频文件 
    # video_folder = os.path.join(root_path, "")
    for file_name in os.listdir(video_folder):
        detection_result_list=[]
        video_path = os.path.join(video_folder, file_name)
        print(f"Processing video folder : {video_folder}")
        i = 0
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            continue
        while True:
            ret, frame = cap.read()
        
            # 检查是否成功读取到帧
            if not ret:
                break
            
            # 在这里处理帧
            detection_result = mediapipe_model.get_video_landmarker(frame,landmarker)
            detection_result_list.append({
                "detection_result":detection_result,
                "frame":frame,
                "frame_index":i
            })
            i = i + 1
        file_name_no_ext,_ = os.path.splitext(os.path.basename(file_name))
        data_list.append({
            "file_id":file_name_no_ext,
            "file_name":file_name,
            "detection_result":copy.deepcopy(detection_result_list)
        })
    
    return data_list
    

def process_data_list(data,rules):
    # 确定好每个视频所需要的规则
    rule_video_map = dict()
    for item in rules:
        for video_id in item["video_ids"]:
            rule_video_map[video_id] = item['rule']

    # 筛选出最好的视频帧
    video_frame_score_list = dict()
    # 处理数据，按照规则处理视频
    for item in data:
        file_id = item['file_id']
        detection_result = item['detection_result']
        video_rule = rule_video_map.get(file_id)
                
        if file_id not in video_frame_score_list:
            video_frame_score_list[file_id] = []
        # 数据中的每一项和规则进行计算
        for detection_item in detection_result:
            pose_landmarks = detection_item["detection_result"]
            frame = detection_item["frame"]
            image_height, image_width, _ = frame.shape
            frame_index = detection_item["frame_index"]
            frame_rule_score = []
            for rule_group in video_rule:   
                group_score = []             
                for group_item in rule_group:
                    id = group_item['id']
                    points = group_item['point']
                    p1,p2,p3 = 0,0,0
                    a,b,c = 0,0,0

                    if len(points)==2:
                        a = points[0]
                        b = points[1]        
                    elif len(points)==3:
                        a = points[0]
                        b = points[1]
                        c = points[2]
                    p1 = [pose_landmarks[a].x * image_width,
                        pose_landmarks[a].y* image_height]
                    p2 = [pose_landmarks[b].x* image_width,
                            pose_landmarks[b].y* image_height]
                    p3 = [pose_landmarks[c].x* image_width,
                            pose_landmarks[c].y* image_height]
                    score,val = mediapipe_model.rule_func[group_item['type']](group_item,p1,p2,p3)                    
                    group_score.append({
                        "rule_id":id,
                        "score":score,
                        "val":val,
                        "rule":group_item
                    })
                
                frame_rule_score.append(group_score)
            video_frame_score_list[file_id].append({
                "frame_rule_score":frame_rule_score,
                "frame_index":frame_index,
                "frame":frame,
                "pose_landmarks":pose_landmarks
            })
    return video_frame_score_list
            
    # for item in rules:        
    #         for values in rule_result["data"].values():                                 
    #             for valueItem in values:
    #                 total_score = total_score+valueItem['score']
            
    #         if bestScore<=total_score:
    #             bestScore = total_score
    #             bestVal=rule_result
    #             bestVal["frame"] = frame
    #             bestVal["frameIndex"] = i
    #             bestVal["category"] = item['category']

    #         # 按 'q' 键退出
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

        # 释放视频捕获对象
    #     cap.release()

    #     # 关闭所有OpenCV窗口
    #     cv2.destroyAllWindows()
    #     print("best video frame data",bestVal["frameIndex"], bestVal["data"]["limit_angle"],bestVal["data"]["angle"],bestVal["data"]["height"],bestVal["data"]["limit_height"])
    #     result.append(bestVal)
    #     cv2.imshow('Image', bestVal["annotated_image"])
    #     # 等待键盘事件，参数为等待时间（毫秒）
    #     cv2.waitKey(0)  # 等待直到有键盘事件
    #     cv2.destroyAllWindows()
    
    # # 计算得分
    # limit_count=0
    # limit_category_map=dict()
    # no_limit_count = 0
    # deductedScore = 0
    
    # for resultItem in result:
    #     limit_angle = resultItem["data"]["limit_angle"]
    #     limit_height = resultItem["data"]["limit_height"]

    #     if resultItem['category'] not in limit_category_map:
    #         limit_category_map[resultItem['category']]=1
    #         for limitItem in limit_angle:
    #             if limitItem['score'] == 0:
    #                 limit_count = limit_count + 1
    #         for limitItem in limit_height:
    #             if limitItem['score'] == 0:
    #                 limit_count = limit_count + 1
                                        

    #     angle = resultItem["data"]["angle"]
    #     for item in angle:
    #         if item['score'] == 0:
    #             no_limit_count = no_limit_count + 1
        
    #     height = resultItem["data"]["height"]
    #     for item in height:
    #         if item['score'] == 0:
    #             no_limit_count = no_limit_count + 1

    # if limit_count==1:
    #     deductedScore=0.1
    # elif limit_count==2:
    #     deductedScore = 1.1
    # elif limit_count==3:
    #     deductedScore = 2.1
    # elif limit_count>=4:
    #     deductedScore = 3.1
    
    # deductedScore = deductedScore + no_limit_count*0.1

    # print(deductedScore)
    


if __name__ == "__main__":
	main()