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
epochs = 20
lr = 0.03
batch_size = 32
# 'pose_case/bird_photos'
# 'pose_case/bird_photos'
image_path = os.path.join(data_path, 'pose_case/bird_photos')
save_path = './checkpoints/bird_model.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1.数据转换
data_transform = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.Resize((512,512)),
        # transforms.RandomResizedCrop(224), # 随机裁剪
        # transforms.RandomHorizontalFlip(), # 左右翻转
        transforms.ToTensor()# 均值方差归一化
    ])
}

# 2.形成训练集
train_dataset = datasets.ImageFolder(root=os.path.join(image_path),
                                     transform=data_transform['train'])


# 3.形成迭代器
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size,
                                           True)

print('using {} images for training.'.format(len(train_dataset)))

# 4.建立分类标签与索引的关系
cloth_list = train_dataset.class_to_idx
class_dict = {}
for key, val in cloth_list.items():
    class_dict[val] = key
with open('class_dict.pk', 'wb') as f:
    pickle.dump(class_dict, f)

# 自定义损失函数，需要在forward中定义过程
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    # 参数为传入的预测值和真实值，返回所有样本的损失值，自己只需定义计算过程，反向传播PyTroch会自动记录，最好用PyTorch进行计算
    def forward(self, pred, label):
        # pred：[32, 4] label：[32, 1] 第一维度是样本数
        
        exp = torch.exp(pred)
        tmp1 = exp.gather(1, label.unsqueeze(-1)).squeeze()
        tmp2 = exp.sum(1)
        softmax = tmp1 / tmp2
        log = -torch.log(softmax)
        return log.mean()

# 5.加载googlenet模型
model = torchvision.models.googlenet(pretrained=True) # 加载预训练好的googlenet模型

# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
    
# 修改最后一层的全连接层
model.fc = nn.Linear(model.fc.in_features, 4)

# 将模型加载到cpu中
model = model.to(device=device)

# criterion = nn.CrossEntropyLoss() # 损失函数
criterion = MyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器

# 6.模型训练
best_acc = 0 # 最优精确率
best_model = None # 最优模型参数

for epoch in range(epochs):
    model.train()
    running_loss = 0 # 损失
    epoch_acc = 0  # 每个epoch的准确率
    epoch_acc_count = 0  # 每个epoch训练的样本数
    train_count = 0  # 用于计算总的样本数，方便求准确率
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
        # 计算每个epoch正确的个数
        epoch_acc_count += (output.argmax(axis=1) == labels.to(device).view(-1)).sum()
        train_count += len(images)

    # 每个epoch对应的准确率
    epoch_acc = epoch_acc_count / train_count

    # 打印信息
    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % str(running_loss))
    print("训练精度为%s" % (str(epoch_acc.item() * 100)[:5]) + '%')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, save_path)

print('Finished Training')

