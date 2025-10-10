#!/usr/bin/env python
# coding: utf-8

# # 训练用的功能

# ## 载入包

# In[63]:


from torchvision import models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ## 导入预训练模型

# In[25]:


def import_maxvit():
    weights = models.MaxVit_T_Weights.IMAGENET1K_V1
    model = models.maxvit_t(weights = weights)
    return model


# ## 修改模型

# In[27]:


def model_change(model,NUM_CLASSES = 6):
    for param in model.parameters():
        param.requires_grad = True
    model.classifier[5] = torch.nn.Linear(in_features=512, out_features=NUM_CLASSES, bias=False)
    return model


# ## 数据集相关

# ### 定义一个数据集对象

# In[71]:


class CRLM_Dataset(Dataset):
        
    def __init__(self, mode, labels:list, images:list):
        self.mode = mode
        self.image_list = images
        self.label_list = labels
        if self.mode == "train":
            preprocess_train = T.Compose([
                T.RandomChoice([
                    T.RandomHorizontalFlip(p=1.0),
                    T.RandomVerticalFlip(p=1.0),
                    T.RandomRotation(degrees=(-180, 180), interpolation=T.InterpolationMode.BILINEAR),
                ]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值和标准差
            ])
            self.process = preprocess_train
            print("process is train")
        else:
            preprocess_test = T.Compose([
                T.ToTensor(),  # operated on original image, rewrite on previous transform.
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # typically from ImageNet
                ])
            self.process = preprocess_test
            print("process is test")
        
    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        image = self.process(image)
        label = self.label_list[index]
        return (image, label)
    
    def __len__(self):
        return len(self.image_list)


# ### 定义一个预加载对象

# In[ ]:


class Pre_load(Dataset):

    def __init__(self,folder):
        self.total = None # a dataframe records label and path
        self.test = None
        self.train = None
        
        # load folder
        file_paths = []
        for root, dirs, files in os.walk(folder):
            for name in files:
                # 获取文件的完整路径并添加到列表中
                file_path = os.path.join(root, name)
                file_paths.append(file_path)
        labels = [i.split("/")[-2] for i in file_paths]
        df_total = pd.DataFrame({"Label":labels,"Path":file_paths})
        df_total["Expand"] = [i.split(".")[-1] for i in df_total["Path"]]
        df_total = df_total[df_total["Expand"].isin(["jpg","tiff","png","tif"])]
        self.total = df_total

    def merge(self,dataset):
        # 将一个新的datas的total添加到当前dataset
        result = pd.concat([self.total, dataset.total], axis=0).reset_index(drop=True)
        self.total = result

    def split(self,test = 0.1):
        # 根据指定的比例将每个类别的标签随机分为test和train
        test_list = []
        train_list = []
        for label in set(self.total["Label"]):
            df_tmp = self.total[self.total["Label"]==label].reset_index(drop=True)
            df_test = df_tmp.sample(frac=test, random_state=1)
            test_list.append(df_test)
            df_train = df_tmp[~df_tmp["Path"].isin(df_test["Path"])]
            train_list.append(df_train)
        self.test = pd.concat(test_list)
        self.train = pd.concat(train_list)

    def export(self,label_dict,batch_size):
        # 根据train和test生成dataloader
        test_set = CRLM_Dataset(mode="test",labels=[label_dict[i] for i in self.test["Label"].to_list()],images=self.test["Path"].to_list())
        train_set = CRLM_Dataset(mode="train",labels=[label_dict[i] for i in self.train["Label"].to_list()],images=self.train["Path"].to_list())
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_dataloader,test_dataloader
        # 并定义相应的预处理过程


# ### 测试

# In[84]:


if __name__ == "__main__":
    ds = Pre_load(folder = "/DATA/pathology/CRLM_class_dataset")
    print(len(ds.total))
    ds_sup = Pre_load(folder="/DATA/pathology/Download")
    ds.merge(ds_sup)
    len(ds.total)
    ds.split()
    print(f"{len(ds.test) = }, {len(ds.train) = }")
    train_dataloader,test_dataloader = ds.export(batch_size=8)
    print(len(test_dataloader))


# ## 训练相关

# ### 训练函数

# In[29]:


# 定义训练循环
def train(model, train_loader, criterion, optimizer,device):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader): # 调用一次train函数相当于1个epoch，遍历一次train_loader
        
        # images, labels = cutmix_or_mixup(images, labels) # 获得增强后的图像和标签
        
        images = images.to(device) # 转移到GPU/CPU上
        labels = labels.to(device)

        optimizer.zero_grad() # 将优化器中所有参数的梯度归零
        outputs = model(images) # 获得输出值，输出值是一个tensor
        loss = criterion(outputs, labels) # 计算损失函数
        # loss.requires_grad = True
        loss.backward() # 进行反向传播
        optimizer.step() # 使用优化器更新模型的参数

        _, predicted = outputs.max(1) # 根据最大概率的类别索引得到预测标签
        # print(predicted)
        total += labels.size(0) # 将标签数量加上一个batch的大小，计算训练样本的总数量
        correct += predicted.eq(labels).sum().item() # 计算训练集上的预测准确率

        running_loss += loss.item()

        # 清理内存
        del images,labels
        torch.cuda.empty_cache()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total # 计算准确率
    
    # 获得当前的学习率
    learning_rate = optimizer.param_groups[0]['lr']

    return train_loss, train_acc, learning_rate


# ### 测试函数

# In[30]:


# 定义测试循环
def validate(model, val_loader, criterion,device):
    model.eval() # 调验证模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader): # 每个epoch遍历所有数据
            images = images.to(device) # 移动到GPU上
            labels = labels.to(device)

            outputs = model(images) # 获得模型的输出
            loss = criterion(outputs, labels) # 计算损失函数

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss += loss.item()

            # 清理内存
            del images,labels
            torch.cuda.empty_cache()

    val_loss = running_loss / len(val_loader) # 计算损失函数
    val_acc = correct / total # 计算acc

    return val_loss, val_acc # 返回损失函数和acc

