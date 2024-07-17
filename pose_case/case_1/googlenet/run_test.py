import torchvision
from torch import nn
import numpy as np
import os
import json
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as models
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

data_path = '../../../../data'

# 从文件中加载模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载googlenet模型
model = torchvision.models.googlenet(pretrained=True) # 加载预训练好的googlenet模型
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
# 修改最后一层的全连接层
model.fc = nn.Linear(model.fc.in_features, 4)
# 将模型加载到cpu中
model = model.to(device=device)
model.load_state_dict(torch.load('./checkpoints/wushu_model.pth'))

# 加载索引与标签映射字典
with open('class_dict.pk', 'rb') as f:
    class_dict = pickle.load(f)

# 数据变换
data_transform = transforms.Compose(
    [transforms.Resize((512,512)),
     
     transforms.ToTensor()])

# 图片路径
# 'pose_case/bird_photos/Cockatoo/030.jpg'
img_path = os.path.join(data_path,'pose_case/wushu/gongbu/1.png')

# 打开图像
img = Image.open(img_path)

# 对图像进行变换
img = data_transform(img)

plt.imshow(img.permute(1,2,0))
plt.show()

# 将图像升维，增加batch_size维度
img = torch.unsqueeze(img, dim=0)
img = img.to(device)
# 获取预测结果
result = model(img)

index = result.argmax(axis=1).item()
pred = class_dict[index]
print('【分类索引】：%s【预测结果分类】：%s' % (index, pred))
