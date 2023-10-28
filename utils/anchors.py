import numpy as np


class AnchorBox():
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        #特征图的大小
        self.input_shape = input_shape
        #最小的先验框的大小
        self.min_size = min_size
        #最大的先验框的大小
        self.max_size = max_size

        self.aspect_ratios = []
        # 将输入的1，2转换为[1, 1, 2, 1/2] 1,2,3转换为[1, 1, 2, 1/2, 3, 1/3]
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        # --------------------------------- #
        #   获取输入进来的特征层的宽和高
        #   比如38x38
        # --------------------------------- #
        layer_height    = layer_shape[0]
        layer_width     = layer_shape[1]
        # --------------------------------- #
        #   获取输入进来的图片的宽和高
        #   比如300x300
        # --------------------------------- #
        img_height  = self.input_shape[0]
        img_width   = self.input_shape[1]

        box_widths  = []
        box_heights = []
        # --------------------------------- #
        #   self.aspect_ratios一般有两个值
        #   [1, 1, 2, 1/2]
        #   [1, 1, 2, 1/2, 3, 1/3]
        # --------------------------------- #
        # 更加公式计算出来的宽高
        
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # --------------------------------- #
        #   获得所有先验框的宽高1/2
        # --------------------------------- #
        box_widths  = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # --------------------------------- #
        #   每一个特征层对应的步长 layer_width特征图的宽度
        #    原图到特征图的缩放比例
        # --------------------------------- #
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # --------------------------------- #
        #   生成网格中心
        # --------------------------------- #
        # np.linspace：这是NumPy库中的一个函数，用于生成等间距的数值序列。
        # 获取每个网格的中心点
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        #得到二维网格
        centers_x, centers_y = np.meshgrid(linx, liny)
        #展平
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
       
        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        #anchor_boxes中每个元素表示当前所在锚框的中心坐标 x和y
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)

        # 目前维度为 特征图像素的个数X2
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))

        # 获得先验框的左上角和右下角 box_widths为一半的宽度 box_heights为一半的高度
        # xmin
        anchor_boxes[:, ::4]    -= box_widths
        #ymin
        anchor_boxes[:, 1::4]   -= box_heights
        #xmax
        anchor_boxes[:, 2::4]   += box_widths
        #ymax
        anchor_boxes[:, 3::4]   += box_heights

        # --------------------------------- #
        #   将先验框变成小数的形式
        #   归一化
        # --------------------------------- #
        #将xmin xmax 归一化
        anchor_boxes[:, ::2]    /= img_width
        #将ymin ymax 归一化
        anchor_boxes[:, 1::2]   /= img_height
        # 当前维度为 (特征图像素Xnum_anchors)X 4 4是xmin ymin xmax ymax
        anchor_boxes = anchor_boxes.reshape(-1, 4)
        
        # 进行一个clap 防止值超出0 或者1之间
        # min max 用于限制范围
        # minimum 用于限制最小值 按位置比较两个数组的元素，并返回最小值
        # maximum 用于限制最大值 按位置比较两个数组的元素，并返回最大值
        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)

        return anchor_boxes

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_vgg_output_length(height, width):
    #8次卷积 5次池化 后面三次卷积不进行池化 第6次设置步长为2 后面两次步长为1但不填充
    filter_sizes    = [3, 3, 3, 3, 3, 3, 3, 3]
    padding         = [1, 1, 1, 1, 1, 1, 0, 0]
    stride          = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        # 300x300 -> 150x150 -> 75x75 -> 38x38 -> 19x19 -> 10x10 -> 5x5 -> 3x3
        # 高的计算方式为 (height + 2*padding - filter_size) // stride + 1
        # 宽的计算方式为 (width + 2*padding - filter_size) // stride + 1
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    #取最后六个特征层的大小
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]
    
# 输入图像的大小为300x300 anchors_size 有7个
def get_anchors(input_shape = [300,300], anchors_size = [30, 60, 111, 162, 213, 264, 315]):

    # 获得每个特征层的大小
    feature_heights, feature_widths = get_vgg_output_length(input_shape[0], input_shape[1])
    #宽高比为 1, 1, 2, 1/2, 3, 1/3
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]     
    anchors = []

    for i in range(len(feature_heights)):
        #先初始化一个AnchorBox类在调用生成先验框
        anchor_boxes = AnchorBox(input_shape, anchors_size[i], max_size = anchors_size[i+1], 
                    aspect_ratios = aspect_ratios[i]).call([feature_heights[i], feature_widths[i]])
        anchors.append(anchor_boxes)
    # 将所有的先验框进行堆叠
    #按照列方向进行堆叠
    #anchors大小为 8732X4
    anchors = np.concatenate(anchors, axis=0)
    return anchors



