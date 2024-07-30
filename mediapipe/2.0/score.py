import pose_data
import rule_data
from rule_data import Config
from typing import Dict,List
import rule_func


def get_video_frame_score_map(pose_data_file_path,rule_file_path):
    # file_path = './pose_data.json'
    pose_list = pose_data.load_json(pose_data_file_path)

    # file_path = './snlhq_rule.json'
    config:Config = rule_data.load_json(rule_file_path)
    
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
        video_id = pose_item.id    
        if video_id  not in video_rules_map:
            print(f"未找到视频: {video_id} 的配置的规则")
            continue
        video_rule = video_rules_map[video_id]
        rule_list_list = video_rule.rule
        
        if video_id not in video_frame_score_map:
            video_frame_score_map[video_id] = []
        
        keypoints = pose_item.keypoints
        frame_rule_score = []
        for frame_index, frame_points in enumerate(keypoints):
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
            
            video_frame_score_map[video_id].append({
                "frame_index":frame_index,
                "frame_rule_score":frame_rule_score,                                
                "frame_points":frame_points
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
                                              
                if r.score_type=='1' and s<0.8:
                    standard_scores[c] = 1
                elif r.score_type=='2' and s<0.8:
                    err_flage = True
                print("类别：%s, 规则：%s, 值: %s, 得分：%s" % (c,r,scoreItem['val'],s))   

        if err_flage:           
            err_scores_count = err_scores_count+1

    print("标准：%d, 动作错误：%d" % (len(standard_scores),err_scores_count))


def score():
    score_map = get_video_frame_score_map('./pose_data.json','./snlhq_rule.json')
    best_map = get_best_frame_data(score_map)
    print_score(best_map)

# 使用示例
def main():
    score()

if __name__ == "__main__":
	main() 
