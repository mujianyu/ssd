import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes    = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape                                                                                                                                                                                                                                                           
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale
        '''
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))：这一行代码计算新的图像形状 new_shape，
        通过将图像原始形状 image_shape 缩放到与输入形状 input_shape 的最小边相匹配。这是为了确保图像可以适应输入大小，同时保持纵横比。

        offset = (input_shape - new_shape)/2./input_shape：这一行代码计算图像有效区域相对于图像左上角的偏移 offset。
        它表示了在将图像调整为 input_shape 大小后，有效区域相对于图像原始左上角的偏移。offset 的计算是通过将 input_shape - new_shape 并除以 2 * input_shape 来获得的。

        scale = input_shape/new_shape：这一行代码计算缩放比例 scale，用于将坐标和目标框的尺寸缩放到与新的图像大小 new_shape 相匹配。

        box_yx = (box_yx - offset) * scale：这一行代码将目标框的中心坐标 box_yx 减去 offset 并乘以 scale，以调整目标框的坐标，以适应新的图像大小。

        box_hw *= scale：这一行代码将目标框的宽度和高度 box_hw 乘以 scale，以调整目标框的尺寸，以适应新的图像大小。
        '''

        box_mins    = box_yx - (box_hw / 2.)#预测框的左上角
        box_maxes   = box_yx + (box_hw / 2.)#预测框的右下角
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        #将预测框的左上角和右下角进行堆叠 boxes shape为（8732，4）
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        #将预测框的左上角和右下角乘以原图的宽高，得到预测框相对于原图的位置
        #boxes *[w,h,w,h] shape为（8732，4）
        #boxes 的4个值分别为预测框的左上角和右下角的y，x坐标
        
        return boxes
    
    #解码的过程
    def decode_boxes(self, mbox_loc, anchors, variances):

        # 获得先验框的宽与高
        anchor_width     = anchors[:, 2] - anchors[:, 0]
        anchor_height    = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x  = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y  = 0.5 * (anchors[:, 3] + anchors[:, 1])

        #预测框的中心预测结果
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        #center_x的预测结果乘以先验框的宽，再加上先验框的中心点，得到预测框的中心点
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        #center_y的预测结果乘以先验框的高，再加上先验框的中心点，得到预测框的中心点
        decode_bbox_center_y += anchor_center_y
        
        # mbox_loc 回归的（x,y,w,h）
        
        # 预测框的宽高的预测结果
        decode_bbox_width   = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width   *= anchor_width
        #bbox_width的预测结果的exp乘以先验框的宽，得到预测框的宽
        decode_bbox_height  = torch.exp(mbox_loc[:, 3] * variances[1])
     
        decode_bbox_height  *= anchor_height
        #bbox_height的预测结果的exp乘以先验框的高，得到预测框的高

        # 预测框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 预测框的位置
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), dim=-1)
        #torch.cat dim等于-1，就是最后一维，也就是列 
        #decode_bbox的shapw为（8732，4）
      
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox
    
    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances = [0.1, 0.2], nms_iou = 0.3, confidence = 0.5):
        #---------------------------------------------------#
        #   :4是回归预测结果
        #---------------------------------------------------#
        mbox_loc        = predictions[0]
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        mbox_conf       = nn.Softmax(-1)(predictions[1])

        results = []
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            results.append([])
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = mbox_conf[i, :, c] 
                #   第c类取出大于门限的框
                c_confs_m   = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    #如果c类存在大于门限的框

                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = decode_bbox[c_confs_m]#从解码后的预测框中取出大于门限的框
                    confs_to_process = c_confs[c_confs_m]#从置信度中取出大于门限的框的置信度
                    '''
                    nms 定义置信度阈值和IOU阈值取值。
                    按置信度score降序排列边界框bounding_box
                    从bbox_list中删除置信度score小于阈值的预测框
                    循环遍历剩余框，首先挑选置信度最高的框作为候选框.
                    接着计算其他和候选框属于同一类的所有预测框和当前候选框的IOU。
                    如果上述任两个框的IOU的值大于IOU阈值,那么从box_list中移除置信度较低的预测框
                    重复此操作，直到遍历完列表中的所有预测框。
                    '''
                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )#进行非极大抑制

                    '''
                    boxes_to_process: 这是一个包含待处理的目标框（或称为边界框）的列表或数组。每个目标框通常由四个坐标值表示，
                    如 (x_min, y_min, x_max, y_max)，分别表示左上角和右下角的坐标。boxes_to_process 包含了所有需要进行非极大值抑制的目标框。
                    confs_to_process: 这是与 boxes_to_process 对应的目标框的置信度分数列表。
                    置信度分数通常是用于衡量模型对于每个目标框中是否包含目标的信心值。
                    通常，confs_to_process 的长度与 boxes_to_process 相同，每个目标框有一个对应的置信度分数。
                    nms_iou: 这是 IoU（Intersection over Union） 的阈值，用于确定何时认为两个目标框之间存在重叠。
                    如果两个目标框的 IoU 大于等于 nms_iou，则它们被认为存在重叠，NMS 将保留置信度较高的那个目标框。

                    按照 confs_to_process 中的置信度分数降序排序 boxes_to_process 中的目标框和对应的置信度。

                    从排序后的目标框列表中选择第一个目标框（即置信度最高的目标框），将其添加到最终的选定目标框列表中。

                    遍历剩余的目标框，对于每个目标框，计算其与已选定目标框列表中的目标框的 IoU（交并比）。

                    如果 IoU 大于等于 nms_iou，则舍弃该目标框，否则将其添加到最终的选定目标框列表中。

                    继续重复步骤3和步骤4，直到处理完所有的目标框。
                    '''

                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #添加一列表示当前类别c
                    '''
                    labels: 这是一个变量或张量，用于存储分配给每个目标框的类别标签。它的形状应该是 (len(keep), 1)，其中 len(keep) 表示经过非极大值抑制（NMS）后保留的目标框数量，
                    而 1 表示每个目标框分配一个类别标签。
                    c: 这是一个整数值，通常用于表示类别的数量。在一些目标检测任务中，
                    常见的类别数包括物体类别如"汽车"、"行人"、"狗"等。c - 1 通常是用于表示背景（background）的类别索引。因此，如果有 c 个类别，通常类别索引从 0 到 c-1，其中 0 表示背景。
                    confs: 这是一个张量，它包含了与每个目标框相关的置信度分数。通常，这些置信度分数用于判断每个目标框属于哪个类别。
                    '''
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    #将预测框的位置、类别、置信度进行堆叠
                    #dim=1表示按列进行堆叠
                    #c_predshape为（n,6）n为预测框的数量 6为预测框的位置、类别、置信度

                    # 添加进result里
                    results[-1].extend(c_pred)#将预测框的位置、类别、置信度添加进result里
                    #results[-1].extend(c_pred)表示将c_pred添加到results的最后一行
             #results的shape为（1，n，6）n为预测框的数量 6为预测框的位置、类别、置信度

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]#将预测框的位置转换为中心点坐标和宽高
                #box_xy为预测框的中心点坐标 shape为（n，2） n为预测框的数量
                #box_wh为预测框的宽高 shape为（n，2） n为预测框的数量
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
                #将预测框的位置转换为相对于原图的位置

        return results
