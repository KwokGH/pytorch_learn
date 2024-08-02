import json
import pose_data
import rule_data
from rule_data import Config
from typing import Dict,List
import rule_func
import cv2
import os
import draw
import mediapipe as mp



def get_video_frame_score_map(pose_list,config):        
    # 以视频为单位，确定某些视频是哪些规则
    video_rules_map: Dict[str, rule_data.Rule]={}
    for item in config.rules:
        for video_id in item.video_ids:           
            video_rules_map[video_id] = item

    # video_pose_map = Dict[str,pose_data.Action] = {}
    # # 获取视频对应的数据
    # for item in pose_list:
    #     video_pose_map[]
    
    # 计算每个视频的每一帧的分值
    video_frame_score_map = {}
    for pose_item in pose_list:    
        video_id = pose_item.index    
        video_label = pose_item.label 
       
        if video_id  not in video_rules_map:
            print(f"未找到视频: {video_id} 的配置的规则")
            continue
        video_rule = video_rules_map[video_id]
        rule_list_list = video_rule.rule
        
        if video_id not in video_frame_score_map:
            video_frame_score_map[video_id] = []
        
        keypoints = pose_item.keypoints
        frame_start_index = pose_item.start
        
        for frame_index, frame_points in enumerate(keypoints):
            if len(frame_points)<32:
                continue
            frame_rule_score = []            
            for rule_list in rule_list_list:                                                       
                group_score = [] 
                for rule_item in rule_list:
                    rule_points = rule_item.point
                    p1 = rule_func.Keypoint(0,0,0)
                    p2 = rule_func.Keypoint(0,0,0)
                    p3 = rule_func.Keypoint(0,0,0)
                    a,b,c = 0,0,0
                    if len(rule_points)==2:
                        a = rule_points[0]
                        b = rule_points[1]        
                    elif len(rule_points)==3:
                        a = rule_points[0]
                        b = rule_points[1]
                        c = rule_points[2]
                    
                    # if frame_points[a].presence<=0.999 or frame_points[b].presence<=0.999:
                    #     continue
                    
                    p1 = rule_func.Keypoint(
                        frame_points[a].x,
                        frame_points[a].y,
                        frame_points[a].z)
                    p2 = rule_func.Keypoint(
                        frame_points[b].x,
                        frame_points[b].y,
                        frame_points[b].z)
                    p3 = rule_func.Keypoint(
                        frame_points[c].x,
                        frame_points[c].y,
                        frame_points[c].z)
                    if c == 0:
                        p3 = rule_func.Keypoint(0,0,0)

                    score,val = rule_func.rule_func[rule_item.type](rule_item,p1,p2,p3)
                    group_score.append({                        
                        "score":score,
                        "val":val,
                        "rule_item":rule_item,
                        "category":video_rule.category,
                    })                
                frame_rule_score.append(group_score)   
            if len(frame_rule_score)<=0:
                print("没有统计到",video_id)   
            video_frame_score_map[video_id].append({
                "frame_index":frame_index,
                "frame_rule_score":frame_rule_score,                                
                "frame_points":frame_points,
                "frame_start_index":frame_start_index,
                "video_label":video_label
            })
        
    return video_frame_score_map

def get_best_frame_data(score_map:dict):
    best = dict()
    for video_id,score_item in score_map.items():
        best[video_id]=dict()
        frame_temp = dict()
        rule_count = len(score_item[0]['frame_rule_score'])
        for i in range(rule_count):
            frame_temp[i]=0           
        for idx,frameItem in enumerate(score_item):
            if len(score_item)>20:                         
                if idx<3 or idx >len(score_item)-3:
                    continue
            # print(len(frameItem['frame_rule_score']))
            for index,scores in enumerate(frameItem['frame_rule_score']):                                
                temp_score = 0                
                               
                for scoreItem in scores:                                        
                    temp_score = temp_score + scoreItem['score']

                if frame_temp[index]<=temp_score:                   
                    frame_temp[index] = temp_score
                    best[video_id][index] = frameItem                                    
    return best

def print_score(best_score:dict):
    standard_scores = dict()
    err_scores_count = 0
    for idx,fileItem in best_score.items():
        print("视频编号：%s. " % (idx))
        err_flage = False
        for idx2,frameItem in fileItem.items():
            frame_index = frameItem["frame_index"]
            print("帧索引：%s. " % (frame_index)) 
            scores1 = frameItem['frame_rule_score']
            scores2 = scores1[idx2]            

            for scoreItem in scores2:
                s = scoreItem['score']
                r = scoreItem['rule_item']
                c = scoreItem['category']
                                              
                if r.score_type=='1' and s<0.85:
                    standard_scores[c] = 1
                elif r.score_type=='2' and s<0.85:
                    err_flage = True
                print("类别：%s, 规则：%s, 值: %s, 得分：%s" % (c,r,scoreItem['val'],s))   

        if err_flage:           
            err_scores_count = err_scores_count+1

    print("标准：%d, 动作错误：%d" % (len(standard_scores),err_scores_count))

def print_score_file(best_score:dict):
    standard_scores = dict()
    err_scores_count = 0
    data={}
    data["result"] = ""
    video_frame_map = {}
    for idx,fileItem in best_score.items():               
        data[idx]={
            "视频编号":idx,
        }
        err_flage = False
        standard_Flage = False
        for idx2,frameItem in fileItem.items():
            frame_file_name = str(idx) 
            frame_index = frameItem["frame_index"]
            frame_start_index = frameItem["frame_start_index"]
            frame_points = frameItem["frame_points"]
            frame_file_name = frame_file_name+"_"+ str(frame_index)
            video_label = frameItem["video_label"]

            frame_index_key="帧索引"+str(frame_index)
            data[idx][frame_index_key]=[]            
            scores1 = frameItem['frame_rule_score']
            scores2 = scores1[idx2]
            
            score_points=[]
            for scoreItem in scores2:
                s = scoreItem['score']
                r = scoreItem['rule_item']
                c = scoreItem['category']
                val = scoreItem['val']
                                              
                if r.score_type=='1' and s<0.85:
                    standard_scores[c] = 1
                    standard_Flage=True
                elif r.score_type=='2' and s<0.85:
                    err_flage = True
                data[idx][frame_index_key].append({
                    "类别":c,
                    "规则":r,
                    "值":val,
                    "得分":s,                    
                })              
                score_points.append({
                        "point":r.point,                        
                        "类别":c,
                        "score":s,
                        "value":val, 
                        "video_label":video_label 
                    })
            is_koufen = False
            if standard_Flage or err_flage:
                is_koufen = True
            video_frame_map[frame_file_name] = {
                "index":frame_start_index+frame_index,
                "points":frame_points,
                "score_points":score_points,
                "is_koufen":is_koufen
            }                  

        if err_flage:           
            err_scores_count = err_scores_count+1
    return len(standard_scores),err_scores_count,data,video_frame_map

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, rule_data.RuleDetail):
            return obj.to_dict()
        
        return super().default(obj)
def score(pose_list,config,result_save_path):   
    score_map = get_video_frame_score_map(pose_list,config)
    best_map = get_best_frame_data(score_map)
    
    standard_count,err_count,data,video_frame_map = print_score_file(best_map)    

    miss_count = config.count-len(pose_list)    
    result_str = "标准出错：%d, 动作错误：%d, 漏做：%d" % (standard_count,err_count,miss_count)
    data["result"] =result_str
    # 将字典保存到 JSON 文件
    p = os.path.join(result_save_path,'result_data.json')
    with open(p, 'w',encoding='utf-8') as json_file:
        json.dump(data, json_file,cls=CustomEncoder,ensure_ascii=False, indent=4)
    
    return video_frame_map


def save_video_frame(video_frame_map,video_path,frame_save_path):
    video_capture = cv2.VideoCapture(video_path)
    # 获取视频的帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for file_name, frame_value in video_frame_map.items():
        frame_index = frame_value["index"]
        frame_points = frame_value["points"]
        frame_score_val = frame_value["score_points"]
        is_koufen = frame_value["is_koufen"]
        # 检查帧索引是否有效
        if frame_index < 0 or frame_index >= total_frames:
            print("视频帧不合法")
            continue
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # 读取指定帧
        ret, frame = video_capture.read()
        if not ret:
            print("读取帧出错")
            continue
     
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        annotated_image = draw.draw_landmarks_on_image(mp_image.numpy_view(), frame_points,frame_score_val,is_koufen)
        # 保存帧到本地文件
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        frame_path = os.path.join(frame_save_path, 'frame_{}.jpg'.format(file_name))
        cv2.imwrite(frame_path, annotated_image)
    
    # 释放视频捕捉对象
    video_capture.release()
    print("保存成功")

