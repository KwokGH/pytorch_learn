import score
import argparse
import pose_data
import rule_data
from rule_data import Config

ap = argparse.ArgumentParser()
ap.add_argument("-p","--pose",required=True,help="pose data file")
ap.add_argument("-r","--rule",required=True,help="rule file")
ap.add_argument("-v","--video",required=False,help="video file")
ap.add_argument("-vf","--video_frame",required=False,help="video frame save folder")
ap.add_argument("-vr","--video_result",required=False,help="video frame result json")


args = vars(ap.parse_args())
# python run.py -p action_2.json -r snlhq_rule.json -v "C:\Users\user\Downloads\\2.mp4" -vf "C:\Users\user\Downloads\\2" 
def main():
    pose_file = args["pose"]
    rule_file = args["rule"]
    print("pose_file:",pose_file)
    print("rule_file:",rule_file)
    pose_list = pose_data.load_json(pose_file)
    config:Config = rule_data.load_json(rule_file)

    result_save_path = args["video_result"]
    video_frame_map = score.score(pose_list,config,result_save_path)
    video_path = args["video"]
    frame_save_path = args["video_frame"]
    score.save_video_frame(video_frame_map,video_path,frame_save_path)

if __name__ == "__main__":
	main() 