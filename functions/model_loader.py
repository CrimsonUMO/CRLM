#!/usr/bin/env python
# coding: utf-8

# # 加载并导出模型

# ## 载入包

# In[12]:


import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import torch

# import .utils as utils
from . import utils

from pathlib import Path
from PIL import Image
from tiatoolbox import data,logger
logger.setLevel(logging.ERROR)
logger.propagate = False
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.tools import stainnorm
from tiatoolbox.utils.visualization import overlay_prediction_mask
from torchvision import transforms

def load_model(pth_path = '/home/lijinghua/DigitalPath/model/final/early_stoped_test.pth',
               pt_path = None,
               HE_norm_method = "Vahadane"):
    # ## HE标准化
    
    # In[7]:
    def HE_normalize(method = "Vahadane"):
        stain_normalizer = stainnorm.get_normalizer(method) # one of “reinhard”, “custom”, “ruifrok”, “macenko” or “vahadane”
        target_image = data.stain_norm_target()
        stain_normalizer.fit(target_image)
    
    
    # ## 设置标签列表
    
    # ## 分类器配置
    
    # In[4]:
    
    
    size = [224, 224] #要改成224*244的
    wsi_ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1}],
        patch_input_shape=size,
        stride_shape=size,
    )
    
    torch.cuda.is_available()
    model_path = pth_path
    external_model = torch.load(model_path,map_location=torch.device('cpu'))
    if pt_path:
        external_model.load_state_dict(torch.load(pt_path))
    external_model.eval()
    
    # 定义预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    
    # 重新定义函数
    def infer_batch(
        model: torch.nn.Module,
        batch_data: np.ndarray or torch.Tensor,
        on_gpu: bool,
    ) -> np.ndarray:
        """Model inference on a batch of images."""
        model.eval() # 变为评估模式
        device = "cuda" if on_gpu else "cpu" # 判断是否放到GPU上
    
        imgs = batch_data # 接受一组batch_data
    
        imgs = imgs.to(device).type(torch.float32) # 放到CPU或GPU上
    
        with torch.inference_mode():
            outputs = model(imgs) # 获得输出值，输出值是一个tensor
            _, predicted = outputs.max(1) # 根据最大概率的类别索引得到预测标签
    
        return predicted.cpu().numpy() # 将输出的预测标签转换为numpy数组，放到CPU上
    
    
    def postproc_func(output: np.ndarray) -> np.ndarray:
        """Pre-processing function."""
        return output
    
    
    def preproc_func(img: np.ndarray) -> np.ndarray:
        """Post-processing function."""
        try:
            img = stain_normalizer.transform(img.copy()) # HE标准化
        except:
            img = img.copy()
        # pil_image = Image.fromarray(img) # 将输入图像转换为PIL图像对象
        transformed = transform(img) # 对图像进行转换处理
        return np.array(transformed) # 得到处理后的图像
    
    
    # 将方法添加至外部模型
    external_model.infer_batch = infer_batch
    external_model.preproc_func = preproc_func
    external_model.postproc_func = postproc_func
    predictor = PatchPredictor(model=external_model, batch_size=64)

    return predictor

# 加载tia模型
def load_tia_model(model_name,batch_size = 64):
    predictor = PatchPredictor(pretrained_model=model_name,batch_size=batch_size)
    return predictor

# 获得推理结果
def tia_predict(predictor,HE_file,wsi_ioconfig,re_folder = f"/DATA/pathology/Download/tia_tmp",ON_GPU = True):
    
    if os.path.exists(re_folder):
        shutil.rmtree(re_folder)
    
    result_path = f"{re_folder}/tia_predict.csv"
    pred_path = f"{re_folder}/tia_predict.png"
    pred_map_path = f"{re_folder}/tia_predict_numpy.png"
    
    # 预测WSI分类
    wsi_output = predictor.predict(
        imgs=[Path(HE_file)],
        mode="wsi",
        merge_predictions=False,
        ioconfig=wsi_ioconfig,
        return_probabilities=False,
        save_dir=re_folder,
        on_gpu=ON_GPU,
    )
    torch.cuda.empty_cache()
        
    return wsi_output

# 可视化配置
class vis_opt:
    def __init__(self,res,unit,label_color_dict,dpi):
        self.res = res
        self.unit = unit
        self.label_color_dict = label_color_dict
        self.dpi = dpi

# 输出推理结果
def tia_output(predictor,wsi_output,HE_file,pic_path,vis_opt):

    # 保存预测图
    thum = utils.thumbnail(HE_file,to_array = True,lv=vis_opt.res)
    pred_map = predictor.merge_predictions(
        HE_file,
        wsi_output[0],
        resolution=vis_opt.res,
        units=vis_opt.unit,
    )
    if thum.shape[:2] != pred_map.shape[:2]:
        thum = Image.fromarray(thum)
        thum = thum.resize(pred_map.shape[1::-1])
        thum = np.array(thum)
    overlay = overlay_prediction_mask(
        thum,
        pred_map,
        alpha=0.5,
        label_info=vis_opt.label_color_dict,
        return_ax = True
    )
    plt.savefig(pic_path,bbox_inches='tight',dpi = vis_opt.dpi)
    print("tia output saved")
    torch.cuda.empty_cache()
    