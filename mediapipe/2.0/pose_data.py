import json
from typing import List

class Keypoint:
    def __init__(self, x, y, z, visibility, presence):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

class Action:
    def __init__(self, id, label, start, end, keypoints: List[List[Keypoint]]):
        self.id = id
        self.label = label
        self.start = start
        self.end = end
        self.keypoints = keypoints

    @classmethod
    def from_dict(cls, data):
        keypoints = [[Keypoint(**kp) for kp in kp_list] for kp_list in data['keypoints']]
        return cls(data['id'], data['label'], data['start'], data['end'], keypoints)


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        actions = [Action.from_dict(action) for action in data]
    return actions

# 使用示例
def main():
    file_path = 'pose_data.json'
    actions = load_json(file_path)

    # 打印加载的数据
    for action in actions:
        print(f"ID: {action.id}, Label: {action.label}, Start: {action.start}, End: {action.end}")
        for kp_list in action.keypoints:
            for kp in kp_list:
                print(f"  Keypoint - x: {kp.x}, y: {kp.y}, z: {kp.z}, visibility: {kp.visibility}, presence: {kp.presence}")

if __name__ == "__main__":
	main()        
