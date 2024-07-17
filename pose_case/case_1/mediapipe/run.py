import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import image_classifier

import matplotlib.pyplot as plt

# model_path = '/absolute/path/to/efficientnet_lite0_int8_2.tflite'

# BaseOptions = mp.tasks.BaseOptions
# ImageClassifier = mp.tasks.vision.ImageClassifier
# ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# data_path = '../../../../data'

# base_options = BaseOptions(model_asset_path=model_path)

# options = ImageClassifierOptions(
#     base_options=BaseOptions(model_asset_path='/path/to/model.tflite'),
#     max_results=5,
#     running_mode=VisionRunningMode.IMAGE)

# mp_image = mp.Image.create_from_file('D:\\project\\machineL\\data\\pose_case\\wushu\\gongbu\\1.png')

# with ImageClassifier.create_from_options(options) as classifier:
#     classification_result = classifier.classify(mp_image)
#     print(classification_result)


