import argparse
import json
import mediapipe_model
import os
import cv2

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
    landmarker = mediapipe_model.initModel(model_path)    
    rules = list(config["rules"])
    result = []
    for item in rules:    
        # video_folder = os.path.join(root_path, item['path'])
        video_path = os.path.join(root_path, item['path'])
        print(f"Processing video: {video_path}")
        i = 0
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            continue
        # 读取视频的每一帧
        bestVal=dict()
        bestScore=0
        while True:
            ret, frame = cap.read()
            
            # 检查是否成功读取到帧
            if not ret:
                break

            i = i + 1
            # if i%2!=0:
            #     continue
            # if i!=26:
            #     continue

            # 在这里处理帧
            total_score = 0
            rule_result = mediapipe_model.process_rules(frame,item,landmarker)
            
            for values in rule_result["data"].values():                                 
                for valueItem in values:
                    total_score = total_score+valueItem['score']
            
            if bestScore<=total_score:
                bestScore = total_score
                bestVal=rule_result
                bestVal["frame"] = frame
                bestVal["frameIndex"] = i
                bestVal["category"] = item['category']

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放视频捕获对象
        cap.release()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        print("best video frame data",bestVal["frameIndex"], bestVal["data"]["limit_angle"],bestVal["data"]["angle"],bestVal["data"]["height"],bestVal["data"]["limit_height"])
        result.append(bestVal)
        cv2.imshow('Image', bestVal["annotated_image"])
        # 等待键盘事件，参数为等待时间（毫秒）
        cv2.waitKey(0)  # 等待直到有键盘事件
        cv2.destroyAllWindows()
    
    # 计算得分
    limit_count=0
    limit_category_map=dict()
    no_limit_count = 0
    deductedScore = 0
    
    for resultItem in result:
        limit_angle = resultItem["data"]["limit_angle"]
        limit_height = resultItem["data"]["limit_height"]

        if resultItem['category'] not in limit_category_map:
            limit_category_map[resultItem['category']]=1
            for limitItem in limit_angle:
                if limitItem['score'] == 0:
                    limit_count = limit_count + 1
            for limitItem in limit_height:
                if limitItem['score'] == 0:
                    limit_count = limit_count + 1
                                        

        angle = resultItem["data"]["angle"]
        for item in angle:
            if item['score'] == 0:
                no_limit_count = no_limit_count + 1
        
        height = resultItem["data"]["height"]
        for item in height:
            if item['score'] == 0:
                no_limit_count = no_limit_count + 1

    if limit_count==1:
        deductedScore=0.1
    elif limit_count==2:
        deductedScore = 1.1
    elif limit_count==3:
        deductedScore = 2.1
    elif limit_count>=4:
        deductedScore = 3.1
    
    deductedScore = deductedScore + no_limit_count*0.1

    print(deductedScore)
    


if __name__ == "__main__":
	main()