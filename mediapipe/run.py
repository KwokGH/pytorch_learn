import argparse
import json
import mediapipe_model
import os
import cv2
import copy
import concurrent.futures

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
    video_folder = "D:\project\machineL\data\mediapipe\shaonian\\temp"# "D:\project\machineL\data\mediapipe\\07\\07"# #os.path.join(root_path, "")
    # 重命名文件
    # i = 0
    # for file_name in os.listdir(video_folder):      
    #     video_path = os.path.join(video_folder, file_name)
    #     file_name_no_ext,ext = os.path.splitext(os.path.basename(file_name))
    #     new_video_path = os.path.join(video_folder, str(i)+ext)
    #     i = i+1
    #     print(new_video_path)
    #     os.rename(video_path, new_video_path)
    # return 
    # 提取规则
    rules = list(config["rules"])
    # 确定好每个视频所需要的规则
    rule_video_map = dict()
    for item in rules:
        for video_id in item["video_ids"]:
            rule_video_map[video_id] = item
    
    # 处理视频获取数据
    data = getData(model_path,video_folder,rule_video_map)

    # 处理数据，获取分数
    result = process_data_list(data,rule_video_map)

    best = dict()
    for file_id,video_item in result.items():
        best[file_id]=dict()

        frame_temp = dict()
        # 规则有几组
        rule_count = len(video_item[0]['frame_rule_score'])
        
        for i in range(rule_count):
            frame_temp[i]=0

        for frameItem in video_item:                        
            for index,scores in enumerate(frameItem['frame_rule_score']):                                
                temp_score = 0
                                
                for scoreItem in scores:                    
                    # temp_score.append((scoreItem['val'],scoreItem['score']))
                    temp_score = temp_score + scoreItem['score']

                if frame_temp[index]<temp_score:
                    # print("temp_score",temp_score)
                    frame_temp[index] = temp_score
                    best[file_id][index] = frameItem
                # if best[file_id][frameItem['frame_index']][index]<temp_score:
                #     best[file_id][frameItem['frame_index']][index] = temp_score


                # print("index:",frameItem['frame_index'],"score:",temp_score)
    # 以视频为单位整理分数 
    output_frame_folder=os.path.join(root_path,"frame")
    for idx,fileItem in best.items():
        print("视频编号：%s. " % (idx))
        for idx2,frameItem in fileItem.items():
            frame_index = frameItem["frame_index"]
            print("视频帧索引：%s. " % (frame_index)) 
            scores1 = frameItem['frame_rule_score']   
            scores2 = scores1[idx2]
            frame = frameItem['frame'] 
            frame_filename = os.path.join(output_frame_folder, f'video_{idx}_frame_{frame_index}.jpg') 
            print(frame_filename)
            cv2.imwrite(frame_filename, frame) 
            for scoreItem in scores2:
                print("类别：%s, 规则：%s, 值: %s, 得分：%s" % (scoreItem['category'],scoreItem['rule'],scoreItem['val'],scoreItem['score']))   
                
                # if len(scores)>=i+1:                    
                #     for scoreItem in scores[i]:
                #         print("类别：%s, 规则：%s,得分：%s " % (scoreItem['category'],scoreItem['rule'],scoreItem['score']))   
            # for scores in frameItem['frame_rule_score']:
            #     for scoreItem in scores:
            #         print("类别：%s, 规则：%s,得分：%s " % (scoreItem['category'],scoreItem['rule'],scoreItem['score']))                    
            # cv2.imshow('Image',frameItem['frame'])
            # cv2.waitKey(0)  # 等待直到有键盘事件
            # cv2.destroyAllWindows()
    # 以种类为单位整理分数
    # category_map = dict()
    # for idx,fileItem in best.items():
    #     for idx2,frameItem in fileItem.items():
    #         max_score = 0
    #         group_score_list = []
    #         for scores in frameItem['frame_rule_score']:
    #             # 由于规则组中的所有规则都会对一个帧进行计算，所以分种类取得时候仅仅取分最高得那一组                
    #             group_score=0                
    #             for scoreItem in scores:
    #                 group_score = group_score+scoreItem['score']
    #                 group_score_list.append({
    #                     "score":scoreItem,
    #                     # "frame":frameItem['frame'],
    #                 })
                
    #             if max_score<group_score:
    #                 max_score = group_score
    #                 if scoreItem['category'] not in category_map:
    #                     category_map[scoreItem['category']] = []
    #                 category_map[scoreItem['category']] = group_score_list


    # print(len(category_map["mabu"]),len(category_map["gongbu"]),len(category_map["tantui"]),len(category_map["dengtui"]))
    # 动作规格扣分， 出现四个规格错误扣3.1分，最多扣3.1分
    # spec_count_map = dict()
    # # 动作错误扣分，累计扣分
    # err_count_map = dict()
    # for key,item in category_map.items():
    #     for scoreItem in item:
    #         scoreInfo = scoreItem['score']
    #         if "score_type" not in scoreInfo['rule']:
    #             continue
            
    #         score_type = scoreInfo['rule']['score_type']
    #         score = scoreInfo['score']
         
    #         if score_type=='1':
    #             print("spec",score)
    #             if score!=1.0:
    #                 if key not in spec_count_map:
    #                     spec_count_map[key] = 1                    
    
            
    #         if score_type=='2':
    #             print("err",score)
    #             if score!=1.0:                   
    #                 if key not in err_count_map:
    #                     err_count_map[key] = 1
    #                 else:
    #                     err_count_map[key] = err_count_map[key]+1
    
    
    # print(spec_count_map)
    # print(err_count_map)


    

                             



def getData(model_path,video_folder,rule_video_map):
      # 加载模型
    landmarker = mediapipe_model.initModel(model_path)   
    data_list = []

    # 遍历读取视频，解析出视频中每一帧的坐标
    # 遍历视频文件 
    listdir = os.listdir(video_folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=34) as executor:
        futures = []
        for file_name in listdir:
            # 如果视频没配置规则，则不对视频处理
            file_name_no_ext,_ = os.path.splitext(os.path.basename(file_name))
            if file_name_no_ext not in rule_video_map:
                continue
            
            futures.append(executor.submit(process_video, data_list, video_folder,file_name,landmarker))
        
        # 等待所有线程完成
        concurrent.futures.wait(futures)
        for future in futures:
            if future.exception() is not None:
                print(f"Error occurred: {future.exception()}")
        
    
    return data_list

def process_video(data_list,video_folder,file_name,landmarker):
    detection_result_list=[]
    video_path = os.path.join(video_folder, file_name)
    if os.path.exists(video_path)==False:
        print(f"video not found: {video_path}")
        return

    print(f"Processing video : {video_path}")
    i = 0
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    while True:
        ret, frame = cap.read()
    
        # 检查是否成功读取到帧
        if not ret:
            break
        
        # 在这里处理帧
        detection_result = mediapipe_model.get_video_landmarker(frame,landmarker)
        if len(detection_result)<=0:
            continue
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

def process_data_list(data,rule_video_map):    
    # 筛选出最好的视频帧
    video_frame_score_list = dict()
    # 处理数据，按照规则处理视频
    for item in data:
        file_id = item['file_id']
        detection_result = item['detection_result']
        if file_id not in rule_video_map:
            continue
        rule_video_map_item = rule_video_map.get(file_id)
        video_rule = rule_video_map_item['rule']
        category = rule_video_map_item['category']
                
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
                        pose_landmarks[a].y* image_height,
                        0]
                    p2 = [pose_landmarks[b].x* image_width,
                            pose_landmarks[b].y* image_height,
                            0]
                    p3 = [pose_landmarks[c].x* image_width,
                            pose_landmarks[c].y* image_height,
                            0]
                    if c == 0:
                        p3 = [0,0,0]
                    score,val = mediapipe_model.rule_func[group_item['type']](group_item,p1,p2,p3)                    
                    group_score.append({                        
                        "score":score,
                        "val":val,
                        "rule":group_item,
                        "category":category,
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