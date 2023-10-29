import colorsys
import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from nets.ssd import SSD300
from utils.anchors import get_anchors
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import BBoxUtility

warnings.filterwarnings("ignore")

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的config里面的num_classes、
#   model_path和classes_path参数的修改
#--------------------------------------------#
class SSD(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/ssd_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        #---------------------------------------------------------------------#
        "input_shape"       : [300, 300],
        #-------------------------------#
        #   主干网络的选择 vgg
        #-------------------------------#
        "backbone"          : "vgg",
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化ssd
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)#获得类别名和类别数
        self.anchors                        = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(torch.FloatTensor)#获得先验框
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes                    = self.num_classes + 1#加上背景类，一共21类
        
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]#hsv_tuples是一个列表，列表中的每个元素是一个元组，元组中的元素是浮点数
        #hsv是一种颜色空间，H代表色调，S代表饱和度，V代表亮度
        #  为每一个类分配一个颜色
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        #list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))的作用是
        # 1.将hsv_tuples中的每个元素传入colorsys.hsv_to_rgb函数中
        # 2.将colorsys.hsv_to_rgb函数的返回值作为列表的元素
        # 3.将列表的元素作为一个列表返回
        # 4.将返回的列表赋值给self.colors
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #将颜色值转换为RGB格式
      
    

        self.bbox_util = BBoxUtility(self.num_classes)#初始化BBoxUtility类
        self.generate()#载入模型
        
        show_config(**self._defaults)#打印模型的参数

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net    = SSD300(self.num_classes, self.backbone)#初始化SSD300类
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #加载预训练的模型
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        #开启评估模式 不计算梯度
        self.net    = self.net.eval()
    
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])#获得图片的高和宽
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。 
        #---------------------------------------------------------#
        #expand_dims 的作用是把数组形状变为(1, 3, 300, 300)
        # preprocess 减去一个means
        # np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)的作用是
        # 1.将image_data转换为float32类型
        # 2.将image_data的维度转换为(3, 300, 300)
        # 3.将image_data的维度转换为(1, 3, 300, 300)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------#
            #   转化成torch的形式
            #---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            #outputs 包括了loc和conf
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            results  = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                return image

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
    
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image







    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    





