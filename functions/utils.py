#!/usr/bin/env python
# coding: utf-8

# # Functions

# ## 载入包

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os
import pandas as pd
import pyvips
from matplotlib.colors import Normalize
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import CubicSpline


# ## 生成缩略图

# In[8]:


def thumbnail(wsi_path,save_path = None,lv = None,to_array = False, verbose = True, scale_bar = True):
    slide = openslide.OpenSlide(wsi_path)
    if not lv:
        lv = slide.level_count//2
    if verbose:
        print(f"selected level is: {lv}")
    w,h = slide.level_dimensions[lv]
    thum = slide.get_thumbnail((w,h))

    if scale_bar:
        factor = 2**lv
        MPP = 0.5*factor
        pix_num = 1000/MPP # 5000微米对应的像素量
        bar_length = int(np.mean(pix_num))
        slide_with_bar = np.array(thum).copy()
        loc_point = np.array(slide_with_bar.shape[:2])//20
        slide_with_bar[loc_point[0]:int(loc_point[0]*1.1),loc_point[0]:loc_point[0]+bar_length,:] = np.array([0,0,0])
        thum = slide_with_bar.copy()
            
    if to_array and isinstance(thum, Image.Image):
        thum = np.array(thum)
    if not to_array and isinstance(thum, np.ndarray):
        thum = Image.fromarray(thum)

    if save_path:
        thum = np.array(thum)
        cv2.imwrite(save_path,cv2.cvtColor(thum,cv2.COLOR_RGB2BGR))
    return thum


# ## 多图展示

# In[6]:


def show_side_by_side(img_list,col = None,title = None,save_path = None,show = True, dpi = 300) -> None:
    """Helper function to shows images side-by-side."""
    plt.ioff()
    length = len(img_list)
    if not col:
        col = length
    row = length // col + (length % col > 0)
    plt.figure(figsize=(col*2,row*2))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i,img in enumerate(img_list):
        plt.subplot(row, col, i+1)
        plt.imshow(img)
        plt.axis("off")
        if title:
            if i < len(title):
                plt.title(title[i])
    if save_path:
        plt.savefig(save_path, dpi = dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    plt.ion()

def wsi_rot(src_path,dst_path):
    vips_image = pyvips.Image.new_from_file(src_path, access='sequential')
    rotated_image = vips_image.rot180()
    rotated_image.tiffsave(dst_path, compression='jpeg', pyramid=True, tile=True, depth='onetile')

def confirm_continue(log = None):
    # 打印日志
    print(f"{log}")
    
    # 获取用户输入
    user_input = input("键入 'Y' 确认继续: ").strip().upper()
    
    # 检查用户输入
    if user_input == 'Y':
        print("确认成功，继续执行...")
        pass
    else:
        print("未确认，程序终止。")
        sys.exit(1)

def rewrite_tiff(src_path,dst_path):
    slide = openslide.OpenSlide(src_path)
    width, height = slide.dimensions
    cropped = slide.read_region((0, 0), 0, (width, height))
    sut_slide = pyvips.Image.new_from_array(cropped)
    sut_slide.tiffsave(dst_path,compression = 'jpeg',pyramid = True,tile = True,depth = 'onetile')

def overlap(img1,img2,alpha = 127):
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)
        
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')
    
    overlap = np.array(img1).copy()
    overlap[:,:,3] = alpha
    overlap = Image.fromarray(overlap)
    new = img2.copy()
    new.paste(overlap, (0, 0), overlap)
    return new

# crop thumbnail
def crop_thum(thum,crop):
    # crop: a list of cut ratio, in order of left,right,top,bottom
    width, height = thum.size
    left,right,sky,land=crop
    x0 = int(width*left)
    y0 = int(height*sky)
    x1 = int(width*(1-left-right)) + x0
    y1 = int(height*(1-sky-land)) + y0
    cropped = thum.crop((x0, y0, x1, y1))
    return cropped

# 切片裁剪
def slide_cut(src_path,
              dst_path,
              crop = (0,0,0,0),# remove%
             ):
    slide = openslide.OpenSlide(src_path)
    width, height = slide.dimensions
    left,right,sky,land=crop
    
    x0 = int(width*left)
    y0 = int(height*sky)
    w = int(width*(1-left-right))
    h = int(height*(1-sky-land))
    # print(x0,y0,w,h)

    cropped = slide.read_region((x0, y0), 0, (w, h))
    slide.close()
    sut_slide = pyvips.Image.new_from_array(cropped)
    sut_slide.tiffsave(dst_path,compression = 'jpeg',pyramid = True,tile = True,depth = 'onetile')

def find_file_path(folder,key = "HE"):
    files = [i for i in os.listdir(folder) if key in i]
    if len(files)==1:
        file = files[0]
        HE_path = f"{folder}/{file}"
    else:
        HE_path = [f"{folder}/{i}" for i in files]
    return HE_path

def wsi2patches(wsi_path, output_folder, num=None, edge=224, format='jpg', box=None):
    """
    将一个WSI切割为若干指定尺寸的patches，并保存为PNG或JPG格式。

    参数:
        wsi_path (str): WSI文件的路径。
        output_folder (str): 保存patch的文件夹路径。
        num (int, optional): 生成的patch数量。默认为空（无限制）。
        edge (int, optional): 生成的patch的边长，默认为224像素。
        format (str, optional): 保存的patch的格式，默认为'jpg'。
        box (tuple, optional): WSI中的局部选区，格式为(x, y, width, height)，默认为空（处理整个WSI）。
    """
    # 预处理：检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开WSI文件
    try:
        wsi = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"无法打开WSI文件 {wsi_path}，错误信息: {e}")
        return
    
    # 获取WSI的宽度和高度
    wsi_width, wsi_height = wsi.dimensions

    # 如果box参数不为空，则限制处理区域
    if box:
        x_start, y_start, box_width, box_height = box
        x_end = x_start + box_width
        y_end = y_start + box_height
    else:
        x_start, y_start, x_end, y_end = 0, 0, wsi_width, wsi_height

    # 初始化变量
    patch_count = 0
    saved_patches = 0

    # 循环切分patch
    for x in tqdm(range(x_start, x_end, edge)):
        for y in range(y_start, y_end, edge):
            # 确定当前patch的边界
            patch_x_start = x
            patch_y_start = y
            patch_x_end = min(x + edge, x_end)
            patch_y_end = min(y + edge, y_end)

            # 计算实际patch的宽度和高度
            patch_width = patch_x_end - patch_x_start
            patch_height = patch_y_end - patch_y_start
            if not patch_width==edge or not patch_height==edge:
                print(patch_width)
                continue

            # 读取patch区域
            patch = wsi.read_region((patch_x_start, patch_y_start), 0, (patch_width, patch_height))

            # 转换为RGB模式（避免透明通道问题）
            patch = patch.convert("RGB")

            # 保存patch到指定文件夹
            patch_filename = f"{patch_x_start}_{patch_y_start}_{patch_x_end}_{patch_y_end}.{format}"
            patch_filepath = os.path.join(output_folder, patch_filename)
            patch.save(patch_filepath, format=format.upper())

            # 更新计数器
            patch_count += 1
            saved_patches += 1

            # 如果num非空且已生成的patch数量达到限制，则退出循环
            if num is not None and saved_patches >= num:
                break
        
        # 提前退出外层循环
        if num is not None and saved_patches >= num:
            break

    # 输出完成提示
    print(f"已完成从WSI路径 {wsi_path} 切割patches，共生成 {saved_patches} 个patch。")

def rgb_to_hex(rgb):
    """
    将RGB色彩转换为十六进制色彩字符串。

    参数:
    r (int): 红色分量，范围0到255。
    g (int): 绿色分量，范围0到255。
    b (int): 蓝色分量，范围0到255。

    返回:
    str: 十六进制色彩字符串，格式为'#RRGGBB'。
    """
    r,g,b = rgb
    # 检查输入是否在有效范围内
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB值必须在0到255之间")
    
    # 格式化为十六进制字符串
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def overmap(wsi_path,csv_path,lv = 4):
    pred_dict = pd.read_csv(csv_path)
    
    slide = openslide.OpenSlide(wsi_path)
    output_shape = slide.level_dimensions[lv][::-1]
    fx = np.array(output_shape) / np.array(slide.dimensions[::-1])
    
    output = np.zeros(list(output_shape), dtype=np.float32)
    
    for idx, bound in enumerate(pred_dict['coordinates']):
        bound = eval(bound)
        prediction = int(pred_dict['predictions'][idx])
        tl = np.ceil(np.array(bound[:2]) * fx).astype(np.int32)
        br = np.ceil(np.array(bound[2:]) * fx).astype(np.int32)
        # print(f"{tl = },{br = }")
        output[tl[1] : br[1], tl[0] : br[0]] += prediction
    
    return output

def predmap2color(predmap,color_dict):
    RGB_array = np.zeros([predmap.shape[0], predmap.shape[1], 3],dtype=np.uint8)
    for key,value in color_dict.items():
        if key==0:
            continue
        label,rgb = value
        RGB_array[predmap==key] = rgb
    return RGB_array

def calculate_centroid(img):
    
    # 获取前景像素的坐标（假设前景为白色，即值>0）
    y_coords, x_coords = np.where(img == 1)
    
    # 检查是否有前景像素
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("图像中没有前景像素")
    
    # 计算重心
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return (centroid_x, centroid_y)

def point_interpolation(points: np.ndarray, num_interp_points = 5000) -> np.ndarray:
    """
    对点集进行插值操作。
    
    :param points: 输入的点集，必须是一个 numpy.ndarray 对象，形状为 (n, 2)
    :return: 插值后的点集，形状为 (m, 2)
    """
    # 检查类型
    if not isinstance(points, np.ndarray):
        raise TypeError("输入的 points 必须是一个 numpy.ndarray 对象")
    
    # 检查形状
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("输入的 points 必须是一个形状为 (n, 2) 的二维数组")

    # 闭合点集
    if not np.allclose(points[0], points[-1], rtol=1e-15, atol=1e-15):
        # print("首尾点不相同！已自动闭合点集")
        points = np.vstack([points, points[0]])  # 将第一个点添加到末尾
    
    # 分离 x 和 y 坐标
    x = points[:, 0]
    y = points[:, 1]
    
    # 计算累积弧长
    t = np.zeros(len(points))
    for i in range(1, len(points)):
        t[i] = t[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    
    # 确保闭合：将最后一个点与第一个点连接起来
    t[-1] += np.linalg.norm(points[0] - points[-1])
    
    # 创建周期性样条插值
    cs_x = CubicSpline(t, x, bc_type='periodic')
    cs_y = CubicSpline(t, y, bc_type='periodic')
    
    # 在参数范围内均匀采样
    t_interp = np.linspace(0, t[-1], num_interp_points)
    
    # 使用插值函数计算新的 x 和 y 坐标
    x_interp = cs_x(t_interp)
    y_interp = cs_y(t_interp)
    
    # 合并为新的点集
    interp_points = np.column_stack((x_interp, y_interp))

    # 输出
    return interp_points

def closest_points(contour1, contour2):
    """
    找到两个轮廓之间的最近点对。
    
    参数:
        contour1 (np.ndarray): 第一个轮廓。
        contour2 (np.ndarray): 第二个轮廓。
    
    返回:
        tuple: 最近点对 (point1, point2)。
    """
    min_dist = float('inf')
    nearest_points = (None, None)
    
    for pt1 in contour1:
        for pt2 in contour2:
            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_dist:
                min_dist = dist
                nearest_points = (tuple(pt1[0]), tuple(pt2[0]))
    
    return nearest_points


def find_nearest_region_and_connect(img, dist_threshold = 10, thickness=3):
    """
    从面积最大的连通区域出发，逐步连接最近的连通区域，直到所有区域之间的最短距离大于给定阈值。
    
    参数:
        img (np.ndarray): 输入的二值图像（0,1 标记的连通区域）。
        dist_threshold (float): 距离阈值，当所有连通区域之间的最短距离大于此值时停止。
        thickness (int): 连通区域的粗细。
    
    返回:
        np.ndarray: 修改后的图像，所有符合条件的连通区域已连接。
    """
    # 复制输入图像，避免修改原始数据
    img = img.astype(np.uint8).copy()
    # print(img.shape)
    while True:

        # 1. 提取连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # print(f"当前图像含有{num_labels}个连通区域")
        
        if num_labels <= 2:  # 如果连通区域少于等于1个，则结束
            break
        
        # 获取每个连通区域的轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f'当前图像含有{len(contours)}个轮廓')
        
        if len(contours) <= 1:  # 如果没有足够的轮廓，则结束
            break

        # 绘制连通区域
        connected_area = np.zeros_like(img)
        cv2.fillPoly(connected_area, contours, color=1)
        
        # 2. 计算所有连通区域之间的最小距离
        min_dist = float('inf')
        nearest_pair = None
        nearest_point1, nearest_point2 = None, None
        
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                contour1 = contours[i]
                contour2 = contours[j]
                
                # 找到两个轮廓之间的最近点对
                point1, point2 = closest_points(contour1, contour2)
                
                # 计算两点间的欧几里得距离
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_pair = (i, j)
                    nearest_point1, nearest_point2 = point1, point2
        
        # 3. 判断是否满足停止条件
        if min_dist > dist_threshold:
            break
        
        # 4. 连接最近的两个区域
        # print(min_dist)
        cv2.line(img, nearest_point1, nearest_point2, color=1, thickness=thickness)
    return img

def find_tumor_edge(img, gap = 3, num_interp_points = 1000):
    '''
    img：完成连通区域检测和相连之后的图像，np.array，（0,1）
    gap：对轮廓进行下采样以提取轮廓骨架，每隔gap个点保留一个点。默认为3。
    num_interp_points：采用插值算法平滑轮廓。期望插值之后的新点集包含多少个点。
    '''
    # 轮廓检测
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个空白图像用于绘制外部边缘
    external_edges = np.zeros_like(img)
    cv2.drawContours(external_edges, contours, -1, (255), thickness=1)

    # 抽样新的点集
    new_edge = [i[::gap] for i in contours]
    new_edge = [i.reshape(len(i), 2) for i in new_edge]
    new_points = [point_interpolation(i,num_interp_points=len(i)*10) for i in new_edge]
    new_points = [np.expand_dims(i, axis=1).astype(np.int32) for i in new_points]
    
    # 填充多边形
    external_edges = np.zeros_like(img)
    cv2.fillPoly(external_edges, new_points, color=1)

    
    return external_edges

def img_filter(img, method = 'open'):
    '''
    对图像进行预处理
    img：一个（0,1）的二值图像。类型为np.array
    method：指定图块的过滤算法。可选项包括'gaussian'和'open'
    '''
        
    if method == 'gaussian': # 如果是高斯滤波
        filtered = cv2.GaussianBlur(img, (3, 3), 0)
    elif method=='open': # 否则执行开运算
        img_in = cv2.erode(img, np.ones((2, 2), dtype=np.uint8)) #腐蚀
        filtered = cv2.dilate(img_in, np.ones((3, 3), dtype=np.uint8)) #膨胀
    else:
        raise ValueError("method参数必须为'gaussian'或'open'")

    return filtered

def outline(image,mask,color = [0, 0, 255],thickness = 1):
    '''
    color应该是RGB格式的，默认的是蓝色
    返回PIL格式的图片
    '''
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 根据缩略图放大mask
    mask = cv2.resize(mask,(image.shape[1], image.shape[0]),interpolation = cv2.INTER_LANCZOS4)
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

    # 查找并绘制轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color=color, thickness=thickness)

    image_with_edges = Image.fromarray(image)
    return image_with_edges

def number2RGB(num,cmap = 'viridis'):
    '''
    num应该是一个数字向量
    '''
    numbers = np.array(list(set(num)))
    cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=min(numbers), vmax=max(numbers))
    colors = cmap(norm(numbers))
    color_dict = {n:[int(color[0]*255), int(color[1]*255), int(color[2]*255)] for n, color in zip(numbers, colors)}
    return color_dict

def load_XML_roi(anno_path, WSI_path, include_ratio=0.8, output_dir="./patches", verbose=False):
    """
    从XML文件加载ROI，使用OpenSlide读取SVS图像，
    根据掩膜切割图像为224x224图块并保存。

    假设:
    1. XML 文件中的坐标 (X, Y) 对应于 SVS 图像的 Level 0 (最高分辨率) 坐标。
    2. 图块大小固定为 224x224 像素。

    Args:
        anno_path (str): XML注释文件的路径。
        WSI_path (str): SVS格式WSI图像文件的路径。
        include_ratio (float): 图块被保留所需的最小掩膜覆盖比例 (0.0 to 1.0)。
        output_dir (str): 保存图块的目录路径。
        verbose (bool): 是否打印详细信息。
    """
    # --- 1. 打开WSI ---
    try:
        slide = openslide.OpenSlide(WSI_path)
        if verbose:
            print(f"Opened WSI: {WSI_path}")
            print(f"WSI dimensions (level 0): {slide.dimensions}")
            print(f"WSI level count: {slide.level_count}")
    except OpenSlideError as e:
        print(f"Error opening WSI file {WSI_path}: {e}")
        return
    except Exception as e:
         print(f"Unexpected error opening WSI file {WSI_path}: {e}")
         return

    # 获取第0层（最高分辨率）的尺寸
    img_w, img_h = slide.dimensions # 注意：dimensions返回 (width, height)

    # --- 2. 读取XML注释 ---
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    try:
        tree = ET.parse(anno_path)
    except ET.ParseError as e:
        print(f"Error parsing XML file {anno_path}: {e}")
        slide.close()
        return
    except FileNotFoundError:
        print(f"XML file not found: {anno_path}")
        slide.close()
        return
    except Exception as e:
         print(f"Unexpected error parsing XML file {anno_path}: {e}")
         slide.close()
         return

    root = tree.getroot()
    anno_dict = {}

    for annotation in root.findall('Annotation'):
        # 提取Id
        annotation_id = annotation.get('Id')
        if not annotation_id:
             if verbose:
                print("Warning: Found annotation without Id, skipping.")
             continue

        # 提取Name
        attribute_name = "Unknown"
        attributes = annotation.find('Attributes')
        if attributes is not None:
            for attr in attributes.findall('Attribute'):
                attribute_name = attr.get('Name') or "Unnamed_Attr" # 处理 Name 为 None 的情况
                break # 通常只取第一个名称



        # 提取Vertices中每个X和Y的坐标
        x_list = []
        y_list = []
        regions = annotation.find('Regions')
        
        if verbose:
            print(f'Processing Annotation ID: {annotation_id}, Attribute Name: {attribute_name}, Regions counts: {len(regions.findall("Region"))}')
            
        has_valid_region = False
        if regions is not None:
            for region in regions.findall('Region'):
                vertices = region.find('Vertices')
                region_ID = region.attrib['Id']
                if vertices is not None:
                    temp_x_list = []
                    temp_y_list = []
                    for vertex in vertices.findall('Vertex'):
                        try:
                            x = float(vertex.get('X'))
                            y = float(vertex.get('Y'))
                            # 可选：检查坐标是否在图像范围内
                            # 注意：有时坐标可能略微超出，取决于标注工具，这里可以放宽或严格处理
                            # if 0 <= x < img_w and 0 <= y < img_h:
                            temp_x_list.append(x)
                            temp_y_list.append(y)
                            # else:
                            #     if verbose:
                            #         print(f"Warning: Vertex ({x}, {y}) might be out of image bounds ({img_w}, {img_h}).")
                        except (ValueError, TypeError) as e:
                            if verbose:
                                print(f"Warning: Skipping invalid vertex data in annotation {annotation_id}: {e}")
                    
                    if len(temp_x_list) >= 3: # 至少需要3个点构成多边形
                        
                        x_list.extend(temp_x_list)
                        y_list.extend(temp_y_list)
                        
                        # add to anno_dict
                        key = f"{annotation_id}_{attribute_name}_{region_ID}"
                        anno_dict[key] = [temp_x_list, temp_y_list]
                        has_valid_region = True

        if not has_valid_region or not x_list or not y_list:
            if verbose:
                print(f"Warning: Annotation {annotation_id}_{attribute_name} has no valid regions or coordinates, skipping.")
            continue

    # --- 3. 处理每个ROI ---
    total_saved = 0

    for key, (x_list, y_list) in anno_dict.items():
        if not x_list or not y_list:
            continue

        # --- 4. 生成全尺寸掩膜 (在ROI区域内) ---
        # 为了效率，我们不创建整个WSI大小的掩膜，而是为每个ROI确定一个边界框
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        
        # 确定ROI的边界框
        roi_x_min = max(0, int(np.floor(np.min(x_array))))
        roi_y_min = max(0, int(np.floor(np.min(y_array))))
        roi_x_max = min(img_w, int(np.ceil(np.max(x_array))))
        roi_y_max = min(img_h, int(np.ceil(np.max(y_array))))

        if roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
            if verbose:
                print(f"Warning: Annotation {key} has invalid bounding box, skipping.")
            continue

        roi_w = roi_x_max - roi_x_min
        roi_h = roi_y_max - roi_y_min

        if verbose:
             print(f"Processing Annotation {key} - ROI Bounding Box: ({roi_x_min}, {roi_y_min}) to ({roi_x_max}, {roi_y_max})")

        # 创建ROI区域大小的掩膜
        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8) # (height, width)
        # 将全局坐标转换为ROI掩膜内的局部坐标
        points_local = np.array(list(zip(x_array - roi_x_min, y_array - roi_y_min)), dtype=np.int32)
        if len(points_local) >= 3:
            cv2.fillPoly(mask_roi, [points_local], color=1)
        else:
             if verbose:
                 print(f"Skipping annotation {key} as it doesn't form a valid polygon after processing.")
             continue

        # --- 5. 遍历ROI边界框内的224x224图块 ---
        # 计算在ROI边界框内，以224为步长的起始和结束坐标
        start_x_patch = (roi_x_min // 224) * 224
        start_y_patch = (roi_y_min // 224) * 224
        # 计算结束patch的起始坐标（确保不超出图像）
        end_x_patch = min(img_w, ((roi_x_max // 224) + 1) * 224)
        end_y_patch = min(img_h, ((roi_y_max // 224) + 1) * 224)

        for y in range(start_y_patch, end_y_patch, 224):
            for x in range(start_x_patch, end_x_patch, 224):
                # --- 6. 计算图块与掩膜的交集面积 ---
                # 确定图块在ROI掩膜中的对应区域
                patch_in_roi_x1 = max(0, x - roi_x_min)
                patch_in_roi_y1 = max(0, y - roi_y_min)
                patch_in_roi_x2 = min(roi_w, x + 224 - roi_x_min)
                patch_in_roi_y2 = min(roi_h, y + 224 - roi_y_min)

                # 检查是否有重叠区域
                if patch_in_roi_x1 < patch_in_roi_x2 and patch_in_roi_y1 < patch_in_roi_y2:
                    # 在ROI掩膜中裁剪出对应区域
                    patch_mask_roi = mask_roi[patch_in_roi_y1:patch_in_roi_y2, patch_in_roi_x1:patch_in_roi_x2]

                    # 计算该区域内的掩膜覆盖比例
                    patch_area = patch_mask_roi.shape[0] * patch_mask_roi.shape[1]
                    if patch_area > 0:
                        area_ratio = np.sum(patch_mask_roi > 0) / patch_area
                    else:
                        area_ratio = 0.0

                    # --- 7. 筛选并保存 ---
                    if area_ratio > include_ratio:
                        # 从WSI读取图块
                        try:
                            # read_region(location, level, size)
                            # location is (x, y) in level 0 coordinates
                            # size is (width, height)
                            patch_img_pil = slide.read_region((x, y), 0, (224, 224))
                            # 转换为 NumPy 数组 (OpenSlide 返回 PIL Image)
                            patch_img = np.array(patch_img_pil)
                            
                            # OpenSlide 读取的通常是 RGBA，如果需要 RGB 可以转换
                            if patch_img.shape[2] == 4: # Check for alpha channel
                                 patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2BGR)
                            elif patch_img.shape[2] == 3:
                                # 如果已经是RGB，OpenCV和PIL的通道顺序不同 (RGB vs BGR)
                                # 如果后续处理用OpenCV，建议转换
                                patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
                            
                        except Exception as e:
                            if verbose:
                                print(f"Error reading region ({x}, {y}) from WSI: {e}")
                            continue

                        # 生成文件名
                        # 使用原始坐标命名，更直观
                        filename = f"{key}_{x}_{y}.png"
                        filepath = os.path.join(output_dir, filename)
                        if os.path.exists(filepath):
                            continue

                        # 保存图块 (OpenCV expects BGR)
                        success = cv2.imwrite(filepath, patch_img)
                        if success:
                            total_saved += 1
                            if verbose:
                                print(f"Saved patch: {filename} (Overlap: {area_ratio:.2f})")
                        else:
                            if verbose:
                                print(f"Failed to save patch: {filename}")

    # --- 8. 清理 ---
    slide.close()
    print(f"Finished processing. Total annotations: {len(anno_dict)}. Total patches saved: {total_saved}.")