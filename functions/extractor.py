#!/usr/bin/env python
# coding: utf-8

# # 定义一个特征提取器

# ## 载入包

# In[206]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from PIL import Image
from . import utils

# ## 说明
# - 特征的命名方式：类型+位置+特征

# ## 定义类

# In[219]:


class feature_extractor:
    
    def __init__(self,output_df,label_dict):
        self.pred_df = pd.read_csv(output_df)
        self.label_dict = label_dict
        self.pred_matrix = None
        self.ROI = None
        self.feature = {"First_order":None,"Interaction":None,"Spatial":None}
        self.ROI_seq = {}
        self.points_with_label = {}

    def get_matrix(self):
        w = max([eval(coord)[0]//224 for coord in self.pred_df["coordinates"].to_list()])
        h = max([eval(coord)[1]//224 for coord in self.pred_df["coordinates"].to_list()])
        canvas = np.zeros((h+1,w+1))
        for _,item in self.pred_df[["coordinates","predictions"]].iterrows():
            coord,label = item
            y,x = [i//224 for i in eval(coord)][:2]
            canvas[x,y] = label
        canvas = canvas.astype(np.uint8)
        self.pred_matrix = canvas
        return canvas

    def get_tissue_mask(self):
        tissue_mask = np.array(self.pred_matrix != 0).astype('uint8')
        # 先进行高斯滤波
        tissue_mask = cv2.GaussianBlur(tissue_mask, (3, 3), 0)
        # 然后查找轮廓
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros_like(tissue_mask)
        # 根据轮廓填充
        tissue_mask = cv2.fillPoly(canvas, contours, color=1)
        self.tissue_mask = tissue_mask

    def load_roi(self,ROI_path):
        img_roi = Image.open(ROI_path).convert('RGBA')
        array_roi = np.array(img_roi)[:, :, 3]
        h,w = self.pred_matrix.shape
        resized = cv2.resize(array_roi, (w,h), interpolation=cv2.INTER_AREA)
        _, binary_roi = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        binary_roi = binary_roi//255
        self.ROI = binary_roi

    def load_XML_roi(self,path,include_ratio = 0.8,verbose = False):
        # 读取XML注释
        tree = ET.parse(anno_path)
        root = tree.getroot()
        anno_dict = {}
        for annotation in root.findall('Annotation'):
            
            # 提取Id
            annotation_id = annotation.get('Id')
            
            # 提取Name
            attributes = annotation.find('Attributes')
            attribute_name = None
            if attributes is not None:
                for attr in attributes.findall('Attribute'):
                    attribute_name = attr.get('Name')
                    if verbose:
                        print(f'Annotation ID: {annotation_id}, Attribute Name: {attribute_name}')
        
            # 提取Vertices中每个X和Y的坐标
            x_list = []; y_list = []
            regions = annotation.find('Regions')
            if regions is not None:
                for region in regions.findall('Region'):
                    vertices = region.find('Vertices')
                    if vertices is not None:
                        for vertex in vertices.findall('Vertex'):
                            x = float(vertex.get('X'))
                            y = float(vertex.get('Y'))
                            x_list.append(x);y_list.append(y)
            
            # add to anno_dict
            anno_dict[f"{annotation_id}_{attribute_name}"] = [x_list,y_list]
        
        dim_y,dim_x = self.pred_matrix.shape
        
        for key, item in anno_dict.items():
        
            x_list,y_list = item
        
            # 生成全尺寸掩膜
            mask = np.zeros((dim_y * 224, dim_x * 224), dtype=np.uint8)  # height, width
            points = np.array(list(zip(x_list, y_list)), dtype=np.int32)
            cv2.fillPoly(mask, [points], color=1)
        
            # 确定mask边界
            lim_l, lim_t, w, h = cv2.boundingRect(points)
            lim_r = lim_l + w; lim_b = lim_t + h
            
            # 如果需要对齐到224的倍数，请确保不会超过图像尺寸
            lim_l = max(0, lim_l // 224 * 224)
            lim_t = max(0, lim_t // 224 * 224)
            lim_r = min(mask.shape[1], (lim_r // 224 + 1) * 224)
            lim_b = min(mask.shape[0], (lim_b // 224 + 1) * 224)
            
            # 确定位于边界以内的图块
            patch_list = [eval(i) for i in self.pred_df["coordinates"]]  # 考虑避免使用 eval
            patch_list = [
                i for i in patch_list 
                if lim_l <= i[0] < lim_r and lim_t <= i[1] < lim_b  # 检查图块是否至少部分位于边界内
            ]
        
            # 确定图块与ROI的交集面积
            areas = []
            for i in patch_list:
                patch_mask = mask[i[1]:i[1]+224, i[0]:i[0]+224]
                area = np.sum(patch_mask > 0)/(224*224)  # 计算非零像素的数量作为交集面积
                areas.append(area)
        
            # 筛选重叠面积符合要求的图块
            patch_list = [i for i,area in zip(patch_list,areas) if area > include_ratio]
        
            # 映射到pred_matrix大小
            mask_ROI = np.zeros((dim_y, dim_x), dtype=np.uint8)  # height, width
            for i in patch_list:
                x0,y0 = [n//224 for n in i[:2]]
                mask_ROI[y0,x0] = 1
        
            area_seq = {key:mask_ROI}
            self.ROI_seq.update(area_seq)
        print(f"{len(anno_dict)} annotations in all: {', '.join(anno_dict.keys())}")
        
    def split_roi(self,how = "quartile",factor = 5):
        tissue = np.where(self.pred_matrix != 0, 1, 0)
        img = self.ROI.copy()
        kernel = np.ones((factor, factor), dtype=np.uint8)
        img_in = cv2.erode(img, kernel) #腐蚀
        img_ex = cv2.dilate(img, kernel) #膨胀
        area_inner = img_in
        area_outer = img - img_in
        area_surr = img_ex - img # 取外扩的部分差值
        area_norm = np.where(self.pred_matrix != 0, 1, 0)
        area_norm = np.where((area_norm ==1) & (img_ex != 1), 1, 0)

        if how=="quartile":
            dict_ROI = {"inner":area_inner,"outer":area_outer,"surrounding":area_surr,"normal":area_norm}

        self.ROI_seq.update(dict_ROI)

    def find_tumor_region_round(self,method = 'open', dist_threshold = 10, thickness = 3):
        '''
        img：一个（0,1）的二值图像。类型为np.array
        method：指定图块的过滤算法。可选项包括'gaussian'和'open'
        首先还是进行连通区域检测和连接
        然后进行轮廓检测
        对每个轮廓分别进行round填充
        '''

        # 0.准备变量
        save_canvas = np.zeros_like(self.pred_matrix)
        
        # 1.滤波
        tum_label = [key for key,value in self.label_dict.items() if value=='TUM'][0]
        img = np.where(self.pred_matrix == tum_label, 1, 0).astype(np.uint8)
        filtered = utils.img_filter(img)
        
        # 2.连通区域检测+相连
        tum_connected = utils.find_nearest_region_and_connect(filtered, dist_threshold = dist_threshold, thickness = thickness)
        if np.sum(tum_connected)==0:
            raise ValueError('连通区域为空')
        
        # 3.轮廓检测对每个轮廓进行round
        contours, _ = cv2.findContours(tum_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for points in contours:
    
            # 3.1 根据轮廓进行填充
            mask_cont = cv2.fillPoly(np.zeros_like(tum_connected), [points], color=1)

            # 3.2 计算重心
            cx,cy = utils.calculate_centroid(mask_cont)
        
            # 3.3 剔除上下左右均为1的像素
            padded = np.pad(mask_cont, pad_width=1, mode='constant', constant_values=0)
            up = padded[:-2, 1:-1]
            down = padded[2:, 1:-1]
            left = padded[1:-1, :-2]
            right = padded[1:-1, 2:]
            surrounded = (up == 1) & (down == 1) & (left == 1) & (right == 1)
            filtered_img = mask_cont.copy()
            filtered_img[surrounded] = 0
            
            # 3.4 计算极坐标并创建DataFrame
            remaining_points = np.argwhere(filtered_img == 1)
            rows, cols = remaining_points[:, 0], remaining_points[:, 1]
            dx = cols - cx
            dy = cy - rows  # 转换为笛卡尔坐标系
            distance = np.hypot(dx, dy)
            angle_deg = np.degrees(np.arctan2(dy, dx)) % 360  # 0~360度
            
            df = pd.DataFrame({
                'x': cols,
                'y': rows,
                'angle': angle_deg,
                'distance': distance
            })
            
            # 3.5 按角度分组保留最大距离
            bin_width = 2  # 分箱宽度可调
            df['angle_binned'] = (np.round(df['angle'] / bin_width) * bin_width).astype(int) % 360
            # df['angle_rounded'] = np.round(df['angle'],2) % 360
            max_idx = df.groupby('angle_binned')['distance'].idxmax()
            max_df = df.loc[max_idx]
            
            # 3.6 对点进行插补
            sorted_df = max_df.sort_values('angle_binned')
            border_points = sorted_df[['x', 'y']].values.astype(np.int32)
            new_points = utils.point_interpolation(border_points)
            
            # 3.7 OpenCV绘制
            border_img = np.zeros_like(tum_connected)
            if len(new_points) >= 3:
                cv_pts = new_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(border_img, [cv_pts], True, color=1, thickness=1)
            border_img = cv2.fillPoly(border_img, [cv_pts], color=1)
    
            # 3.8 高斯滤波并重新filloply
            sub_img = cv2.GaussianBlur(border_img, (3, 3), 0)
            sub_contours, _ = cv2.findContours(sub_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sub_img = np.zeros_like(sub_img)
            cv2.fillPoly(sub_img, sub_contours, color=1)
            save_canvas += sub_img

        # 4.输出图片
        _, save_canvas = cv2.threshold(save_canvas, 1, 1, cv2.THRESH_TRUNC)

        # 5.更新ROI
        save_canvas = save_canvas & self.tissue_mask
        self.ROI = save_canvas
        return save_canvas # 现在返回的是二值化的边框图像

    def find_tumor_region_connect(self,method = 'open', dist_threshold = 10, thickness = 3, gap = 3, num_interp_points = 1000):
        '''
        使用连通区域检测+轮廓检测确定肿瘤范围。
        '''
        
        # 1.滤波
        tum_label = [key for key,value in self.label_dict.items() if value=='TUM'][0]
        img = np.where(self.pred_matrix == tum_label, 1, 0).astype(np.uint8)
        filtered = utils.img_filter(img)
        
        # 2.连通区域检测+相连
        tum_connected = utils.find_nearest_region_and_connect(filtered, dist_threshold = dist_threshold, thickness = thickness)
        if np.sum(tum_connected)==0:
            raise ValueError('连通区域为空')
            
        # 3.肿瘤区域检测
        mask = utils.find_tumor_edge(tum_connected, gap = gap, num_interp_points = num_interp_points)

        # 4.轮廓检测+填充
        contours, _ = cv2.findContours(tum_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros_like(tum_connected)
        mask = cv2.fillPoly(canvas, contours, color=1)

        # 5.更新ROI
        mask = mask & self.tissue_mask
        self.ROI = mask
        
        return mask

        
    def dilate(self,dst_label = 5,factor = 3, iteration = 5, use_ROI = True):
        '''
        factor: 卷积核的大小。
        dst_label：通常指定的是肿瘤对应的数字标签。
        iteration：膨胀操作的重复次数，当卷积核为3×3时，重复次数即为从TUM出发向外扩展的patch距离。
        '''
        
        # 设定卷积核
        kernel = np.ones((factor, factor), dtype=np.uint8)

        # 设定初始掩膜
        if use_ROI:
            filtered = self.ROI
        else:
            tissue = np.where(self.pred_matrix != 0, 1, 0)
            tum = np.where(self.pred_matrix == dst_label, 1, 0).astype(np.uint8)
            img_in = cv2.erode(tum, kernel)
            filtered = cv2.dilate(img_in, kernel) # 获得过滤后的mask

        dilated_final = cv2.dilate(self.ROI, kernel,iterations = iteration) # 获得膨胀后的mask

        area_all = dilated_final.copy() # 包含肿瘤在内的整个区域
        area_margin = dilated_final - filtered # 仅Margin区域
        # area_surr = img_ex - img_in # 交界区的 - 标记
        # area_norm = np.where(self.pred_matrix != 0, 1, 0) # 非背景的部分
        # area_norm = np.where((area_norm ==1) & (img_ex != 1), 1, 0) # 除去上面以外的部分
        
        area_seq = {"tumor":self.ROI,"margin":area_margin}
        
        self.ROI_seq.update(area_seq)
    
    # 计算一阶特征
    def calculate_first_order(self):
        list_df = []
        
        for name,matrix in self.ROI_seq.items():
        
            # 选取元素
            area = list(self.pred_matrix[matrix==1])
            re = [[c,list(area).count(i)] for i,c in self.label_dict.items()]
            df_re = pd.DataFrame(re,columns=["Label","count"])
            df_re["ratio"] = [i for i in df_re["count"]/len(area)]
            df_re = pd.melt(df_re,id_vars=["Label"],value_vars=["count","ratio"],var_name="Type",value_name="Value")
            df_re["Feature"] = [f"firstorder_{name}_{x}.{y}" for x,y in zip(df_re["Label"],df_re["Type"])]
            df_re = df_re[["Feature","Value"]]
            list_df.append(df_re)
            # break
        df = pd.concat(list_df)
        self.feature["First_order"] = df

    # 计算交互特征
    def calculate_interaction(self):
        '''
        特征将以`interaction_{name}_{c}~{s}`的形式命名。
        其中，`name`是指当前ROI的名称，如"dilated"，`c`是指当前滑动窗口的中间位置图块类别，`s`是指当前滑动窗口其他位置的图块类别。
        例如，特征名称`interaction_dilated_all_DEB~BACK`，表明该特征是基于"dilated"区域提取的，该特征表征了以"DEB"为中心，周围存在的"BACK"的比例。
        '''
        list_df = []
        mask = np.ones(shape=(3,3),dtype=bool)
        mask[1,1] = False
        col_names = ['center'] + list(self.label_dict.values()) + ['position']
        count_df = pd.DataFrame(columns = col_names)
        
        for name,matrix in self.ROI_seq.items():
        
            # 获得指定区域索引
            area_index = np.where(matrix==1)
            row_list = []
            # 遍历本区域
            for i,j in zip(area_index[0],area_index[1]):
                if self.pred_matrix[i,j] == 0:
                    continue
                center = self.label_dict[self.pred_matrix[i,j]]
                B = self.pred_matrix[i-1:i+2,j-1:j+2]
                count_vals = list(B.flatten())
                if len(count_vals)==0:
                    continue
                count_vals.remove(self.pred_matrix[i,j])
                row = [count_vals.count(i) for i in self.label_dict.keys()]
                row = [center] + row + [(i,j)]
                row_list.append(row)
        
            df = pd.DataFrame(row_list,columns=col_names)
            # 保存df
            # 计算summary
            df_summary = df.groupby('center', as_index=False)[list(self.label_dict.values())].mean()
        
            # 格式转换
            df_re = pd.melt(df_summary,id_vars=["center"],value_vars=list(self.label_dict.values()),var_name="sur",value_name="Value")
            df_re["Feature"] = [f"interaction_{name}_{c}~{s}" for c,s in zip(df_re["center"],df_re["sur"])]
            list_df.append(df_re)
        df_summary = pd.concat(list_df)
        df_summary = df_summary[["Feature","Value"]]
        self.feature["Interaction"] = df_summary

    # 计算分布特征
    def calculate_spatial(self,running_folder,method = "DBSCAN",eps = 5,verbose = False, area = 'all', normalize_area = True, target_max=200):
        '''
        area: 指定用于特征提取的区域。默认为'all'，即确定全片范围内的分布特点。可选：'tumor'，即选择filter之后的肿瘤范围；'margin'，即外扩之后的瘤周范围。
        '''
        
        point_list = []; df_list = []
        features = ["clusters","noises","CHS","SC","DBI","size.mean","size.std","wgssk.mean","wgssk.std","extend.mean","extend.std","ballhall","DRI","BRI","BGSS","WGSS"]
        for i in range(1,len(self.label_dict)):
            # colnames = [f"spatial_{method}_{area}_{self.label_dict[i]}_{fea}" for fea in features]
            colnames = [f"spatial_{area}_{method}.{self.label_dict[i]}.{fea}" for fea in features]
            
            # 选取data
            if area=='all':
                y, x = np.where(self.pred_matrix == i)
            elif area=='tumor':
                y, x = np.where((self.pred_matrix == i) & (self.ROI_seq['tumor']==1))
            elif area=='margin':
                y, x = np.where((self.pred_matrix == i) & (self.ROI_seq['margin']==1))

            # 建立dataframe
            df = pd.DataFrame({'y': y,'x': x})
            
            if len(df)<2:
                if verbose:
                    print(f"too few points, skiped")
                continue
                
            if normalize_area:
                if verbose:
                    print("原始数据：")
                    print(df)
                x_span = df['x'].max() - df['x'].min()
                y_span = df['y'].max() - df['y'].min()
                max_span = max(x_span, y_span)
                scaling_factor = target_max / max_span # 计算缩放因子
                df = df.copy()
                df['x'] = (df['x'] - df['x'].min()) * scaling_factor
                df['y'] = (df['y'] - df['y'].min()) * scaling_factor
                if verbose:
                    print("\n缩放后的数据：")
                    print(df)
                
            if verbose:
                print(f"calculating spatial features of {self.label_dict[i]} with {len(df)} points")

        
            # 密度聚类
            if method == "DBSCAN":
                # eps: ε半径参数, min_samples: MinPts参数
                cluster_obj = DBSCAN(eps=eps, min_samples=2).fit(df)
                if verbose:
                    print(f"Executing DBSCAN clustering for {self.label_dict[i]} in {area}, {eps = }")
            elif method == "AP":
                # 亲和传播聚类
                if len(df)>2000:
                    damping = 0.8
                elif len(df) < 650:
                    damping = 0.6
                else:
                    damping = 0.7
                if verbose:
                    print(f"Executing AP clustering for {self.label_dict[i]} in {area}, damping = {damping}")
                cluster_obj = AffinityPropagation(damping=damping).fit(df)
            
            df_points = df.copy().assign(labels=cluster_obj.labels_, tissue=self.label_dict[i])
            point_list.append(df_points)
            
            # 计算特征
            ### Fea 1: n_clusters is the count of clusters
            n_clusters = len(set(cluster_obj.labels_)) - (1 if -1 in cluster_obj.labels_ else 0)
            ### Fea 2: n_noise is the count of noise points
            n_noise = list(cluster_obj.labels_).count(-1)

            # 生成结果字典
            df_points_clean = df_points.query("labels != -1").groupby("labels").filter(lambda x: len(x) > 1)
            result = df_points_clean.groupby('labels').size().reset_index(name='size').to_dict(orient='list')
        
            if len(set(df_points_clean["labels"]))<2:
                CHS,SC,DBI = None, None, None
            else:
                ### Fea 3: calinski_harabasz_score
                ### https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
                CHS = calinski_harabasz_score(df_points_clean.iloc[:,:2],df_points_clean["labels"])
            
                ### Fea 4: silhouette_score
                ### https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
                SC = silhouette_score(df_points_clean.iloc[:,:2],df_points_clean["labels"]) + 1 # to generate Non-negative Matrix
        
                ### Fea 5: davies_bouldin_score
                ### https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
                DBI = davies_bouldin_score(df_points_clean.iloc[:,:2],df_points_clean["labels"])
            
            # 计算总数据量和簇数量
            wgssk, extend, ball_hall, centroids = [], [], [], []
            n_samples = len(df_points_clean)
            n_clusters = len(set(result["labels"]))
            for k, cluster_data in df_points_clean.groupby("labels"):
                if not cluster_data.empty:
                    cluster_data = cluster_data.iloc[:,:2]
                    # 计算质心
                    centroid = cluster_data.mean().values
                    centroids.append(centroid)
            
                    # 计算组内距离的最大值
                    distances = cdist(cluster_data, [centroid], metric='euclidean').flatten()
                    extend.append(distances.max())
            
                    # 计算簇内平方和及其均值
                    wgss = np.sum(distances**2)
                    wgssk.append(wgss)
                    ball_hall.append(wgss / len(cluster_data))
            
            # 更新result字典
            result.update({
                "wgssk": wgssk,
                "extend": extend,
                "ballhallk": ball_hall,
                "centroids": centroids
            })
        
            # 计算指标
            df_result = pd.DataFrame(result)
            
            ### Feature 6/7: size mean/std
            size_mean,size_std = df_result["size"].agg(['mean', 'std'])
            
            ### Feature 8/9: WGSSk mean/std
            wgssk_mean,wgssk_std = df_result["wgssk"].agg(['mean', 'std'])
            
            ### Feature 10/11: Extend mean/std
            extend_mean,extend_std = df_result["extend"].agg(['mean', 'std'])
            
            ### Feature 12: BallHall index
            ballhall = df_result["ballhallk"].mean()
            
            ### Feature 13: DRI Determinant Ratio Index
            if len(df_points_clean)==0:
                DRI = None
            else:
                # # 协方差矩阵
                # cov_matrices = [np.cov(cluster_data.iloc[:,:2], rowvar=False) for k, cluster_data in df_points_clean.groupby("labels")]
                # # 行列式
                # det_values = [np.linalg.det(cov_matrix) for cov_matrix in cov_matrices]
                # # 总协方差矩阵及其行列式
                # total_det = np.linalg.det(np.cov(df_points_clean.iloc[:,:2], rowvar=False))
                # DRI = total_det / np.mean(det_values)

                epsilon = 1e-6  # 微小正则项
                cov_matrices = [
                    np.cov(cluster_data.iloc[:, :2], rowvar=False) + epsilon * np.eye(2)
                    for k, cluster_data in df_points_clean.groupby("labels")
                ]
                det_values = [np.linalg.det(cov_matrix) for cov_matrix in cov_matrices]
                
                # 总协方差矩阵也做相同处理
                total_cov = np.cov(df_points_clean.iloc[:, :2], rowvar=False) + epsilon * np.eye(2)
                total_det = np.linalg.det(total_cov)
                
                DRI = total_det / np.mean(det_values)
                
            ### Feature 14: BRI
            if len(result["centroids"])<2:
                BRI = None
            else:
                # 簇内平均距离
                intra_cluster_distance = [np.sum(cdist(data.iloc[:,:2], data.iloc[:,:2]))/ (2 * (len(data) - 1)) for k, data in df_points_clean.groupby("labels")]
                # 簇间距离
                inter_cluster_distance = np.mean(cdist(result["centroids"], result["centroids"])[np.triu_indices(len(result["centroids"]), k=1)])
                BRI = np.mean(intra_cluster_distance)/inter_cluster_distance
            
            ### Feature 15: 簇间平方和
            if len(result["centroids"])<2:
                BGSS = None
            else:
                BGSS = np.sum(pdist(result["centroids"], metric='euclidean') ** 2)
                
            ### Feature 16: 簇内平方和
            WGSS = np.sum(result["wgssk"])
        
            values = [n_clusters,n_noise,CHS,SC,DBI,size_mean,size_std,wgssk_mean,wgssk_std,extend_mean,extend_std,ballhall,DRI,BRI,BGSS,WGSS]
            df_row = pd.DataFrame({"Feature":colnames,"Value":values})
            df_list.append(df_row)
        
            # break

        if len(point_list)!=0:
            points_all = pd.concat(point_list)
            points_all.to_csv(f"{running_folder}/{method}_{area}.csv",index = False)
            
        if len(df_list)==0:
            print('No features were calculated due to an insufficient number of data points')
        else:
            if verbose:
                print('Features calculation finished')
            df_sum = pd.concat(df_list,axis=0,join='outer')
            if self.feature["Spatial"] is None:
                self.feature["Spatial"] = df_sum
            else:
                self.feature["Spatial"] = pd.concat([self.feature["Spatial"], df_sum], ignore_index=True)
            self.points_with_label.update({self.label_dict[i]:df_points})

    def output(self):
        tmp = self.feature
        re_list = []
        for key,value in tmp.items():
            re_list.append(value)
        if all(item is None for item in re_list):
            raise ValueError("All features are None")
        elif len(re_list)==0:
            raise ValueError('length of re_list is 0')
        else:
            re_total = pd.concat(re_list)
            return re_total


# ## 测试

# In[221]:


if __name__ == "__main__":
    
    # tissues = ["BACK","DEB","LIN","LYM","STR","TUM"]
    tissues = ["BACK","NORM","DEB","TUM","ADI","MUC","MUS","STR","LYM"]
    NUM_CLASSES = len(tissues)
    label_dict = {i:label for label,i in zip(tissues,range(0,NUM_CLASSES))}
    print(label_dict)
    
    ### 初始化
    p = "P16"
    csv_path = f"/DATA/pathology/CRT/{p}/tia_re/tia_predict.csv"
    roi_path = f"/DATA/pathology/ROI/{p}.png"
    anno_path = '/DATA/pathology/CRT/P16/anno.xml'

    extracter = feature_extractor(
        output_df=csv_path,
        label_dict=label_dict)
    
    ### 还原矩阵
    extracter.get_matrix()
    print(extracter.pred_matrix.shape)
    plt.imshow(extracter.pred_matrix)
    
    ### 加载ROI
    # extracter.load_roi(roi_path)
    # plt.imshow(extracter.ROI)
    
    ### ROI分割
    # extracter.split_roi()
   
    ### 膨胀
    extracter.dilate()
    
    ### 加载XML格式的注释文件
    extracter.load_XML_roi(anno_path)
    print(extracter.ROI_seq)
    
    ### 计算一阶特征
    extracter.calculate_first_order()
    extracter.feature["First_order"].head()
    
    ### 计算交互作用
    extracter.calculate_interaction()
    extracter.feature["Interaction"].head()

    ### 计算空间分布
    extracter.calculate_spatial(method = "DBSCAN",verbose = True)
    extracter.calculate_spatial(method = "AP",verbose = True)
    extracter.feature["Spatial"].head()
    
    ### 输出
    re = extracter.output()
    re.head()

