import json
from typing import List

class RuleDetail:
    def __init__(self, type: str, point: List[int], max: float, min: float, desc: str, score_type: str = None):
        self.type = type
        self.point = point
        self.max = max
        self.min = min
        self.desc = desc
        self.score_type = score_type

class Rule:
    def __init__(self, video_ids: List[str], category: str, rule: List[List[RuleDetail]]):
        self.video_ids = video_ids
        self.category = category
        self.rule = rule

    @classmethod
    def from_dict(cls, data):
        rule = [[RuleDetail(**rd) for rd in sub_rule] for sub_rule in data['rule']]
        return cls(data['video_ids'], data['category'], rule)

class Config:
    def __init__(self, name: str, root_path: str, rules: List[Rule]):
        self.name = name
        self.root_path = root_path
        self.rules = rules

    @classmethod
    def from_dict(cls, data):
        rules = [Rule.from_dict(rule) for rule in data['rules']]
        return cls(data['name'], data['root_path'], rules)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        config = Config.from_dict(data)
    return config

# 使用示例
def main():
    file_path = './snlhq_rule.json'
    config = load_json(file_path)
    
    # 打印加载的数据
    print(f"Name: {config.name}")
    print(f"Root Path: {config.root_path}")
    for rule in config.rules:
        print(f"Category: {rule.category}, Video IDs: {rule.video_ids}")
        for sub_rule in rule.rule:
            for rd in sub_rule:
                print(f"  Type: {rd.type}, Points: {rd.point}, Max: {rd.max}, Min: {rd.min}, Desc: {rd.desc}, Score Type: {rd.score_type}")

if __name__ == "__main__":
	main()  