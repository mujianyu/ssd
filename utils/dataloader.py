import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input



class SSDDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, overlap_threshold = 0.5):
        super(SSDDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.num_anchors        = len(anchors)
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)#1.数据增强并对得到box
        #self.annotation_lines[index] 图像文件名和相关信息的文本 input_shape 输入图像形状大小  train 是否运用数据增强
        #此时返回图像的数据shape为（300，300，3）和真实框的坐标（xminyminxmaxymax类别） shape为（num_true_box, 5）
        image_data  = np.transpose(preprocess_input(np.array(image, dtype = np.float32)), (2, 0, 1))#2.对图像进行归一化并将图像的维度换位 (c,w,h)
        # preprocess_input 减去一个均值means 
        # transpose  将数组的维度换位 将channel维度放在第一位
        # 目前image_data 的shape 为（3，300，300） 3 为channel 300 为图像的宽和高
        
        #---------------------------------------------------#
        #   对真实框进行处理
        #---------------------------------------------------#
        # self.input_shape 为输入图像的大小
        if len(box)!=0:
            boxes               = np.array(box[:,:4] , dtype=np.float32)#取真实框的坐标
            # 进行归一化，调整到0-1之间
            #boxes的格式为xminyminxmaxymax
            #将boxes的x坐标归一化
            boxes[:, [0, 2]]    = boxes[:,[0, 2]] / self.input_shape[1]
            #将boxes的y坐标归一化
            boxes[:, [1, 3]]    = boxes[:,[1, 3]] / self.input_shape[0]
            # 对真实框的种类进行one hot处理
            # 描述一下one hot的过程
            # 1.首先将真实框的种类提取出来，然后将其转换为整数类型
            # 2.然后利用np.eye()函数将其转换为one hot编码
            # 3.最后将其与真实框的坐标进行拼接
            # 描述一下one hot的是干嘛的
            # 1.在训练的时候，我们需要将真实框的种类转换为one hot编码，这样才能与预测结果进行对比
            # 2.在预测的时候，我们需要将预测结果转换为真实框的种类，这样才能知道预测的是什么
            # one hot编码的具体是啥
            # 1.假设我们有5个类别，那么one hot编码就是一个长度为5的数组，其中只有一个元素为1，其余元素都为0

            #one_hot_label返回的是一个数组，数组的长度为类别数-1，数组中的元素为0或1，其中只有一个元素为1，其余元素都为0
            #one_hot_label的shape为[num_true_box, num_classes - 1]
            # num_true_box是指真实框的数量
            # 20 列 关于label的信息
            one_hot_label   = np.eye(self.num_classes - 1)[np.array(box[:,4], np.int32)]
            #box的shape为[num_true_box, 5] 4是真实框的坐标，1是真实框的种类
            box             = np.concatenate([boxes, one_hot_label], axis=-1)
            
            #box的shape为[num_true_box, 5 + num_classes - 1]
            # axis=-1是指沿着最后一个轴进行拼接 即沿着列进行拼接
            # num_classes-1是指不包含背景类
        # 3.对box坐标归一化 并得到one_hot_label 
        #此时box的shape为[num_true_box, 5 + num_classes - 1]

        # 调用assign_boxes函数，对真实框进行编码
        # 编码后结果为[num_anchors, 4 + num_classes + 1]
        # 即每一个先验框都有一个真实框与之对应
        # 4是先验框的坐标，1是iou，num_classes是种类，1是是否包含物体
        # 真实框变成了先验框的形式
        # 找到每一个真实框，重合度较高的先验框
        # 先验框的坐标为xminyminxmaxymax

        # 4.为每个先验框分配真实框 使得每个先验框都有一个真实框与之对应 shape为[num_anchors, 4 + num_classes + 1]
        box = self.assign_boxes(box)
        # box的维度为 [num_anchors, 4 + num_classes + 1] 4是先验框的坐标，1是iou，num_classes是种类，1是是否包含物体
        # image_data为归一化后的图像数据 shape为（3，300，300） 值为0-1之间
        #box为先验框的坐标和种类 shape为[num_anchors, 4 + num_classes + 1]
        return np.array(image_data, np.float32), np.array(box, np.float32)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
 
        line = annotation_line.split()# 按照空格


        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        '''
        line[1:]：假设 line 是一个包含文本行数据的列表或数组，line[1:] 是一个切片操作，
        它返回 line 列表中从索引 1 开始到结尾的所有元素。这通常表示 line 的第一个元素不包含坐标信息，而后面的元素包含。 此处第一个信息是图片的位置信息，后面的信息是box的位置信息
        for box in line[1:]：这是一个循环，它遍历 line[1:] 中的每个元素，将每个元素依次赋值给变量 box。
        list(map(int, box.split(',')))：对于每个 box，这一部分执行了以下操作：
        box.split(',')：这是一个字符串分割操作，它将 box 字符串按逗号 , 进行分割，得到一个包含坐标值的字符串列表。其中box 的内容类似于 "xmin,ymin,xmax,ymax,类别"
        map(int, box.split(','))：map 函数将 int 函数应用于 box.split(',') 中的每个分割后的字符串，将它们转化为整数类型。这将返回一个整数类型的迭代器。
        np.array(...)：最外层的 np.array 函数用于将上述操作得到的整数迭代器转化为NumPy数组。
        '''
        #box的shape为[num_true_box, 5] 4是真实框的坐标xmin,ymin,xmax,ymax，1是真实框的种类
        if not random:
            #如果不随机变化，那么就直接resize图像
            scale = min(w/iw, h/ih) #缩放倍数 将图像变成300*300，取最小的缩放倍数
            nw = int(iw*scale) #缩放后的图像宽度
            nh = int(ih*scale) #缩放后的图像高度
            dx = (w-nw)//2#图像在x轴上的偏移量
            dy = (h-nh)//2#图像在y轴上的偏移量
            # //2为了保持图像的中心点坐标为(w/2, h/2)不变

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)#将图像缩放到nw*nh
            new_image   = Image.new('RGB', (w,h), (128,128,128))#创建一个新的图像，大小为w*h，颜色为灰色
            new_image.paste(image, (dx, dy))#将缩放后的图像粘贴到新的图像上，粘贴的起始位置为(dx, dy)
            image_data  = np.array(new_image, np.float32)#将图像转换为数组

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)#打乱box的顺序
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx #将box的x坐标进行缩放，然后加上偏移量dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy #将box的y坐标进行缩放，然后加上偏移量dy
                box[:, 0:2][box[:, 0:2]<0] = 0 #将box的左上角xy坐标小于0的值设为0
                box[:, 2][box[:, 2]>w] = w #将box的右下角x坐标大于w的值设为w
                box[:, 3][box[:, 3]>h] = h#将box的右下角y坐标大于h的值设为h
                box_w = box[:, 2] - box[:, 0]#计算box的宽度
                box_h = box[:, 3] - box[:, 1]#计算box的高度
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)# 新宽高比为图像的原来的宽高比乘以rand(0.7,1.3)/rand(0.7,1.3)  
        scale = self.rand(.25, 2)#缩放倍数为rand(0.25,2)
        if new_ar < 1:#如果新宽高比小于1 w>h
            nh = int(scale*h)#缩放后的图像高度为scale*h 先对高度进行缩放
            nw = int(nh*new_ar)#缩放后的图像宽度为nh*new_ar 再对宽度进行缩放
        else:#如果新宽高比大于1 w<=h
            nw = int(scale*w) #缩放后的图像宽度为scale*w 先对宽度进行缩放
            nh = int(nw/new_ar)#缩放后的图像高度为nw/new_ar 再对高度进行缩放
        image = image.resize((nw,nh), Image.BICUBIC)#将图像缩放到nw*nh

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))#图像在x轴上的偏移量 dx的范围为[0, w-nw]
        dy = int(self.rand(0, h-nh))#图像在y轴上的偏移量 dy的范围为[0, h-nh]
        new_image = Image.new('RGB', (w,h), (128,128,128)) #创建一个新的图像，大小为w*h，颜色为灰色
        new_image.paste(image, (dx, dy))#将缩放后的图像粘贴到新的图像上，粘贴的起始位置为(dx, dy)
        image = new_image#将新的图像赋值给image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5#以0.5的概率进行翻转
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)#如果flip=True则图像水平翻转

        image_data      = np.array(image, np.uint8)#将图像转换为数组
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        #色域变换是指将图像从RGB空间转换到HSV空间，然后对H、S、V三个通道进行变换，最后再转换回RGB空间
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1 #r的shape为[3]
        
        #例如hue=.1, sat=0.7, val=0.4
        #uniform(-1, 1, 3) 生成3个-1到1之间的随机数 乘以[hue=.1, sat=.7, val=.4] + 1 
        # r值的范围为[0.9,1.7,2.2] 
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))#将图像转换到HSV空间上
        #hueshape为[300,300] satshape为[300,300] valshape为[300,300]
        #HSV是一种颜色空间，它将颜色的色调（H），饱和度（S），明度（V）分离开来，这种颜色空间模型常用于计算机视觉中颜色的描述与识别。

        dtype           = image_data.dtype#图像的数据类型
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)#生成一个数组，数组的元素为0-255 r.dtype为r的数据类型 x的shape为[256] x的值为0-255
        lut_hue = ((x * r[0]) % 180).astype(dtype)#lut_hue的shape为[256]，lut_hue的值为0-179 lut_hue的数据类型为dtype
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)#lut_sat的shape为[256]，lut_sat的值为0-255 lut_sat的数据类型为dtype
        #np.clip()函数的作用是将数组中的元素限制在a_min, a_max之间，大于a_max的元素就使得它等于a_max，小于a_min的元素就使得它等于a_min
        # lut_sat 等于 x * r[1]，但是 lut_sat 的值被限制在 0 和 255 之间。
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # lut_val 等于 x * r[2]，但是 lut_val 的值被限制在 0 和 255 之间。
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))#将图像转换到RGB空间上
        #HSV空间的H通道的取值范围为0-179，S和V通道的取值范围为0-255，而RGB空间的取值范围为0-255
        #cv2.LUT()函数的作用是将图像image_data中的每一个像素的值根据lut_hue、lut_sat、lut_val中的值进行映射
        #cv2.merge()函数的作用是将图像的通道进行合并，例如将图像的三个通道合并为一个通道
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)#将图像转换到RGB空间上

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)#打乱box的顺序
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx#将box的x坐标进行缩放，然后加上偏移量dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy#将box的y坐标进行缩放，然后加上偏移量dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]#如果flip=True则将box的x坐标进行翻转 因为图像已经水平反转过了 
            box[:, 0:2][box[:, 0:2]<0] = 0#将box的左上角xy坐标小于0的值设为0
            box[:, 2][box[:, 2]>w] = w#将box的右下角x坐标大于w的值设为w
            box[:, 3][box[:, 3]>h] = h#将box的右下角y坐标大于h的值设为h
            box_w = box[:, 2] - box[:, 0]#计算box的宽度
            box_h = box[:, 3] - box[:, 1]#计算box的高度
            box = box[np.logical_and(box_w>1, box_h>1)]# discard invalid box 
        
        return image_data, box

    def iou(self, box):
        #---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        #---------------------------------------------#
        inter_upleft    = np.maximum(self.anchors[:, :2], box[:2])
        #两图交集的左上角坐标两个框的左上角坐标的最大值
        inter_botright  = np.minimum(self.anchors[:, 2:4], box[2:])
        #两图交集的右下角坐标两个框的右下角坐标的最小值
        

        #宽高等于右下角坐标减去左上角坐标
        inter_wh    = inter_botright - inter_upleft
        #取最大值，如果小于0，那么表示两个框不相交，取0
        inter_wh    = np.maximum(inter_wh, 0)

        #交集的面积
        inter       = inter_wh[:, 0] * inter_wh[:, 1]
        #---------------------------------------------# 
        #   真实框的面积
        #   box的数据为XminYminXmaxYmax的形式
        #---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        #---------------------------------------------#
        #   先验框的面积
        #   先验框的数据为XminYminXmaxYmax的形式
        #---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        #---------------------------------------------#
        #   计算iou
        #---------------------------------------------#
        #交集等于两个和减去并集
        union = area_true + area_gt - inter
        #交并比
        iou = inter / union
        return iou
    
    # 编码的过程 返回的是先验框的坐标和iou
    def encode_box(self, box, return_iou=True, variances = [0.1, 0.1, 0.2, 0.2]):
        #---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        #---------------------------------------------#
        iou = self.iou(box)#计算当前box与所有先验框anchors的iou
        #anchors的shape为8372X4 xminyminxmaxymax
        #num_anchors为先验框的数量 为8372
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        #初始化encoded_box的shape为[num_anchors, 4 + return_iou]
        
        #---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测 box 由这个来预测
        #---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        #---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本 
        #---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        #---------------------------------------------#
        #   利用iou进行赋值 
        #---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        #---------------------------------------------#
        #   找到对应的先验框
        #---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # 1.当前box与所有的anchors计算iou之后，找到iou大于阈值的anchors或者最大的那个

        #---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        #   box的数据是xminyminxmaxymax的形式
        #---------------------------------------------#
        # 对anchor的坐标进行编码
        # box 真实框 xminyminxmaxymax -> xywh
        box_center  = 0.5 * (box[:2] + box[2:])
        '''
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        box[:2] 表示前两个元素，即左上角的坐标。
        box[2:] 表示后两个元素，即右下角的坐标。
        box[:2] + box[2:] 执行了两组坐标的加法，得到的是左上角和右下角坐标之和。
        0.5 * (box[:2] + box[2:]) 对坐标之和进行了按元素的乘法运算，将其除以2，以计算中心坐标。这将给出中心点的 x 和 y 坐标。
        '''
        box_wh      = box[2:] - box[:2]

        '''
        width = xmax - xmin
        height = ymax - ymin
        box[2:] 包含右下角的坐标。
        box[:2] 包含左上角的坐标。
        box[2:] - box[:2] 执行了两组坐标的减法运算，得到的是右下角和左上角坐标之差。这将给出宽度和高度。
        '''
        #---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        #   assigend_anchors的数据是xminyminxmaxymax的形式
        #   assigned_anchors[:, :2] 表示先验框的左上角坐标。每行的前两个元素。
        #   assigned_anchors[:, 2:4] 表示先验框的右下角坐标。每行的后两个元素。
        #---------------------------------------------#

        # anchors的坐标 xminyminxmaxymax -> xywh
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh     = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        
        #------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果（g-d）/d
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2] 
        #------------------------------------------------#
        # 对anchor的坐标进行编码
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center

        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        # ravel是将多维数组降为一维
        return encoded_box.ravel()



    def assign_boxes(self, boxes):
        #---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        #---------------------------------------------------#
        # anchors与box的不同的地方是，前者没有真实框的种类，后者有真实框的种类 前者有背景类，后者没有背景类
        # assignment的shape为[num_anchors, 4 + 21 + 1] 4是先验框的坐标，21是种类，1是是否包含物体

        # boxes的shape为[num_true_box, 5 +num_class-1 ] 4是真实框的坐标，1是真实框的种类,后面的num_class-1是真实框的种类的one hot编码

        assignment          = np.zeros((self.num_anchors, 4 + self.num_classes + 1))#创建一个全为0的数组，shape为[num_anchors, 4 + 21 + 1] 4是先验框的坐标，21是种类，1是是否包含物体
        assignment[:, 4]    = 1.0 #将assignment的第5列的值设为1，即将背景类的概率设为1 第5列是背景类的概率
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        # np.apply_along_axis的作用是将一个函数应用到某个轴上的元素上
        # 将encode_box应用到boxes的每一行上，即对每一个真实框都进行编码
        # np.apply_along_axis(self.encode_box, 1, boxes[:, :4])的shape为[num_true_box, num_anchors * 5] 4是真实框的坐标，1是iou
        
        # 5.encode_box是对真实框进行编码，返回的是能预测各个box的先验框anchors的编码后的坐标和iou
        encoded_boxes   = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        #encoded shape为[num_true_box, num_anchor * 5] 4是真实框的坐标，1是iou
        #num_anchor<=nums_anchor为能预测各个box的先验框anchors的数量（小于iou被舍弃）最大值为8732

        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes   = encoded_boxes.reshape(-1, self.num_anchors, 5)
        
        #---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        #6.每一个先验框重合度最大与真实框的iou  shape为[num_anchors]
        best_iou        = encoded_boxes[:, :, -1].max(axis=0)
        #encoded_boxes[:, :, -1].max(axis=0)的值为每一个先验框重合度最大的真实框的iou
        # best_iou是为了找到每一个先验框重合度最大的真实框
        
        #best_iou的shape为[num_anchors]，即每一个先验框都有一个iou最大的真实框

        #7.每个先验框的重合度最大的真实框的索引 先验框表示真实框 shape为[num_anchors]
        best_iou_idx    = encoded_boxes[:, :, -1].argmax(axis=0)
        #best_iou_idx的shape为[num_anchors]，即每一个先验框都有一个iou最大的真实框的索引 值为0-20

        # best_iou_mask是iou大于0的掩码 
        best_iou_mask   = best_iou > 0
        #best_iou_mask的shape为[num_anchors]，即每一个先验框都有一个iou大于0的掩码 值为True或False

        # best_iou_idx是iou大于0的索引 大小为iou大于0的先验框的数量
        best_iou_idx    = best_iou_idx[best_iou_mask]
        # 8.只保留iou大于0的先验框 
        
        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num      = len(best_iou_idx)

        # 9.将编码后的真实框取出
        encoded_boxes   = encoded_boxes[:, best_iou_mask, :]
        #shape为[num_true_box, assign_num, 4 + 1] 4是先验框的坐标，1是iou
        #assign_num是iou大于0的先验框的数量

        #---------------------------------------------------#
        #   编码后的真实框的赋值
        #---------------------------------------------------#

        # 10.将编码后的真实框赋值给assignment的能与box iou最大的anchors
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        #assignment[:, :4][best_iou_mask]的shape为[assign_num, 4] 4是先验框的坐标
        #assignment[:, :4][best_iou_mask]的值为编码后的真实框的坐标

        #----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        #----------------------------------------------------------#
        
        # 11. 将能与box iou最大的anchors的背景的概率设为0 
        assignment[:, 4][best_iou_mask]     = 0
        # 等于0表示有对应的物体不是背景
        #assignment的shape为[num_anchors, 4 + 21 + 1] 4是先验框的坐标，21是种类，1是是否包含物体
        
        #boxes有24列
        #11. 将能与box iou最大的anchors的种类设为真实框的种类 
        assignment[:, 5:-1][best_iou_mask]  = boxes[best_iou_idx, 4:]
        # boxes的shape为[num_true_box, 5] 4是真实框的坐标，1是真实框的种类
        # boxes[best_iou_idx, 4:]的shape为[assign_num, 21] 



        #----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        #----------------------------------------------------------#
        # 12. 将能与box iou最大的anchors的是否包含物体设为1
        assignment[:, -1][best_iou_mask]    = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

# DataLoader中collate_fn使用
def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    #这行代码的作用是将images转换为tensor类型
    bboxes = torch.from_numpy(np.array(bboxes)).type(torch.FloatTensor)
    return images, bboxes
