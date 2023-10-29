import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
annotation_mode     = 2
classes_path        = 'model_data/voc_classes.txt'
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
VOCdevkit_path  = 'VOCdevkit'
VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
#获得class的名称和数量
classes, _      = get_classes(classes_path)
#获得图片的数量
photo_nums  = np.zeros(len(VOCdevkit_sets))
#获得类别的数量
nums        = np.zeros(len(classes))
def convert_annotation(year, image_id, list_file):
    #year=2007
    #image_id=000001
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')#打开xml文件
    tree=ET.parse(in_file)#解析xml文件
    root = tree.getroot()#获得根节点
    for obj in root.iter('object'):#遍历根节点下的object节点
        difficult = 0 #默认为0
        if obj.find('difficult')!=None:#如果object节点下有difficult节点
            difficult = obj.find('difficult').text#获得difficult节点的值
            #difficult 节点为1时，代表目标难以识别，为0时则相反。
            #在VOC数据集中，目标较小或者遮挡严重的标注为difficult。         
        cls = obj.find('name').text#获得object节点下的name节点的值
        if cls not in classes or int(difficult)==1:#如果name节点的值不在classes中或者difficult节点的值为1
            continue
        cls_id = classes.index(cls)#获得name节点的值在classes中的索引
        xmlbox = obj.find('bndbox')#获得object节点下的bndbox节点
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        # b为目标的左上角和右下角坐标
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))#将目标的坐标和类别写入list_file 前四个为坐标，最后一个为类别
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1 #统计每个类别的数量
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    print("Generate 2007_train.txt and 2007_val.txt for train.")
    type_index = 0#用于记录是2007_train.txt还是2007_val.txt
    for year, image_set in VOCdevkit_sets:#遍历VOCdevkit_sets 
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
        #打开对应的txt文件，获得图片的id 从测试集或者训练集中获得      
        list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        #打开对应的txt文件，用于写入图片的路径和标签 2007_train.txt或者2007_val.txt      
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))#将图片的路径写入list_file
            convert_annotation(year, image_id, list_file)#将图片的路径和目标位置，标签写入list_file
            # list_file每行的内容为图片的路径和目标位置，标签
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)#统计图片的数量
        type_index += 1
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")
    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')#rjust() 方法返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串。
                print("|", end=' ')
            print()

    str_nums = [str(int(x)) for x in nums]#将nums中的数字转换为字符串
    #nums保存的是每个类别的数量
    #str_nums保存的是每个类别的数量的字符串形式
    tableData = [
        classes, str_nums
    ]#tableData为二维数组，第一行为类别，第二行为类别的数量
    #tableData的shape为(2,20)
    colWidths = [0]*len(tableData)#colWidths为一维数组，长度为tableData的长度，每个元素为0 
    # colwidths的shape为(2,)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                #如果tableData[i][j]的长度大于colWidths[i]
                #则将colWidths[i]的值改为tableData[i][j]的长度
                #colWidths[i]的值为tableData[i]中最长的字符串的长度
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)
    if photo_nums[0] <= 500:
        print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")
    if np.sum(nums) == 0:
        print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
 