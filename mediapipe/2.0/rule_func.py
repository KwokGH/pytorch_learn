from math import atan2, degrees
from dataclasses import dataclass
from typing import List
import math
from rule_data import RuleDetail

@dataclass
class Keypoint:
    x: float
    y: float
    z: float

def calculate_angle_3d(p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
    def vector_from_points(p1: Keypoint, p2: Keypoint) -> List[float]:
        return [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]

    def dot_product(v1: List[float], v2: List[float]) -> float:
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

    def vector_magnitude(v: List[float]) -> float:
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    def angle_between_vectors(v1: List[float], v2: List[float]) -> float:
        dot_prod = dot_product(v1, v2)
        mag_v1 = vector_magnitude(v1)
        mag_v2 = vector_magnitude(v2)
        cos_theta = dot_prod / (mag_v1 * mag_v2)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    v1 = vector_from_points(p1, p2)
    v2 = vector_from_points(p3, p2)
    return angle_between_vectors(v1, v2)

def calculate_angle_2d(p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
    # Helper function to create a vector from two points
    def vector_from_points_2d(p1: Keypoint, p2: Keypoint) -> List[float]:
        return [p2.x - p1.x, p2.y - p1.y]
    
    # Helper function to calculate the dot product of two vectors
    def dot_product_2d(v1: List[float], v2: List[float]) -> float:
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    # Helper function to calculate the magnitude of a vector
    def vector_magnitude_2d(v: List[float]) -> float:
        return math.sqrt(v[0]**2 + v[1]**2)
    
    # Helper function to calculate the angle between two vectors
    def angle_between_vectors_2d(v1: List[float], v2: List[float]) -> float:
        dot_prod = dot_product_2d(v1, v2)
        mag_v1 = vector_magnitude_2d(v1)
        mag_v2 = vector_magnitude_2d(v2)
        cos_theta = dot_prod / (mag_v1 * mag_v2)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    # Create vectors from the given points
    v1 = vector_from_points_2d(p1, p2)
    v2 = vector_from_points_2d(p3, p2)
    
    # Calculate and return the angle between the vectors
    return angle_between_vectors_2d(v1, v2)

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


def process_angle_rule(rule_item:RuleDetail,p1,p2,p3):
  score = 0
  max = rule_item.max
  min = rule_item.min
  get = calculate_angle_2d(p1, p2, p3) 

  score = proximity_to_interval(get,min,max)

  return score,get

def process_weight_rule(rule_item:RuleDetail,p1,p2,p3=[0,0,0]):
  score = 0
  max = rule_item.max
  min = rule_item.min
  x1 = p1.x
  x2 = p2.x
  get = x1-x2

  score = proximity_to_interval(get,min,max)

  return score,get

def process_height_rule(rule_item,p1,p2,p3=[0,0,0]):
  score = 0
  max = rule_item.max
  min = rule_item.min
  y1 = p1.y
  y2 = p2.y
  get = y1-y2

  score = proximity_to_interval(get,min,max)

  return score,get

# 规则注册
rule_func = {
   "angle":process_angle_rule,
   "height":process_height_rule,
   "weight":process_weight_rule
}


def main():
    p1 = Keypoint(0.3179285526275635, 0.5936577916145325, 0.02006792463362217)
    p2 = Keypoint(0.3132648766040802, 0.6692636609077454, 0.018931586295366287)
    p3 = Keypoint(0.27381184697151184, 0.7534188628196716, 0.054403096437454224)
    
    angle = calculate_angle_3d(p1,p2,p3)
    print(angle)
    p1 = Keypoint(0.3179285526275635, 0.5936577916145325,0)
    p2 = Keypoint(0.3132648766040802, 0.6692636609077454,0)
    p3 = Keypoint(0.27381184697151184, 0.7534188628196716,0)
    angle = calculate_angle_2d(p1,p2,p3)
    print(angle)

if __name__ == "__main__":
	main() 