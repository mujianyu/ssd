import datetime
import os
import warnings
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.ssd import SSD300
from nets.ssd_training import (MultiboxLoss, get_lr_scheduler,
                               set_optimizer_lr, weights_init)
from utils.anchors import get_anchors
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import ( get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch
from utils.callbacks import LossHistory
warnings.filterwarnings("ignore")

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''

if __name__ == "__main__":
    Cuda = True#是否使用GPU
    seed            = 11#随机种子
    fp16            = False#是否使用fp16半精度训练
    classes_path    = 'model_data/voc_classes.txt'#类别文件的路径
    model_path      = 'model_data/ssd_weights.pth'#预训练权重的路径
    input_shape     = [300, 300]#输入图片的大小
    backbone        = "vgg"#主干特征提取网络的类型
    pretrained      = False#是否使用预训练权重(model_path不为空则pretrained的值无效)
    anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    #anchor的大小根据voc数据集设定的，大多数情况下都是通用的。如果想要检测小物体，
    #可以修改anchors_size，一般调小浅层先验框的大小就行。因为浅层负责小物体检测！
    Init_Epoch          = 0 
    UnFreeze_Epoch      = 200#总训练世代
    Unfreeze_batch_size = 8#解冻训练的batch_size
    Freeze_Train        = False #是否进行冻结训练
    Init_lr             = 2e-3 #模型的最大学习率
    Min_lr              = Init_lr * 0.01#模型的最小学习率，默认为最大学习率的0.01
    optimizer_type      = "sgd" # 当使用SGD优化器时建议设置   Init_lr=2e-3
    momentum            = 0.937#优化器内部使用到的momentum参数
    weight_decay        = 5e-4#权值衰减，可防止过拟合
    lr_decay_type       = 'cos'#学习率衰减策略
    save_period         = 10#多少个epoch保存一次权值
    save_dir            = 'logs'#权值保存的路径
    eval_flag           = True #是否进行评估
    eval_period         = 10 #多少个epoch评估一次
    num_workers         = 4 #多线程读取数据所使用的线程数
    train_annotation_path   = '2007_train.txt' #训练集的路径  训练图片路径和标签
    val_annotation_path     = '2007_val.txt' #验证集的路径 验证图片路径和标签
    seed_everything(seed)#设置随机种子
    ngpus_per_node  = torch.cuda.device_count() #GPU数量
    print("Use %d GPUs for training"%(ngpus_per_node)) #打印GPU数量
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用GPU还是CPU
    local_rank      = 0#默认为0
    rank            = 0#默认为0
    #1.获取类
    class_names, num_classes = get_classes(classes_path)#获取类别和类别数量
    num_classes += 1 #类别数量加1 再加上背景类，一共是num_classes+1类
    #2.获取anchors
    anchors = get_anchors(input_shape, anchors_size) #获取anchors 
    #3.获取模型
    model = SSD300(num_classes, backbone, pretrained)#实例化模型
    weights_init(model)#初始化模型权重
    #4.加载预训练权重

    #根据预训练权重的Key和模型的Key进行加载
    if model_path != '':
        if local_rank == 0: 
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict() #model_dict是模型权重
        pretrained_dict = torch.load(model_path, map_location = device) #pretrained_dict 是预训练权重
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):#如果预训练权重的key在模型权重中，且形状相同
                temp_dict[k] = v#将预训练权重的键值对存入temp_dict
                load_key.append(k)#将预训练权重的键存入load_key
            else:
                no_load_key.append(k)#如果预训练权重的key不在在模型权重且形状不相同，将预训练权重的键存入no_load_key
        model_dict.update(temp_dict)#update是将temp_dict中的键值对更新到model_dict中
        model.load_state_dict(model_dict)#加载权重
        #  显示没有匹配上的Key
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    #5.得到损失函数
    criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)# 获得损失函数        

    scaler = None#混合精度训练器
    model_train     = model.train()#模型转为训练模式
    #6. 读取训练集和验证集
    if Cuda:       
        model_train = torch.nn.DataParallel(model)#多GPU训练
        cudnn.benchmark = True#设置为True，那么每次运行卷积神经网络的时候都会去寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        model_train = model_train.cuda()#将模型转为cuda模式
    with open(train_annotation_path, encoding='utf-8') as f:#读取训练集的txt
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:#读取验证集的txt
        val_lines   = f.readlines()
    num_train   = len(train_lines)#训练集的大小
    num_val     = len(val_lines)  #验证集的大小

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = 0, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = 0, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )#打印配置信息
        #   总训练epoch指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        
        #7.计算步长

        wanted_step = 5e4 # 想要的总步长
        # 计算总步长 判读是否达到建议的总步长
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        # Unfreeze_batch_size =8 
        # UnFreeze_Epoch = 200

        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S') #time_str是当前时间
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str)) # log_dir路径是logs/loss_当前时间
    #8.实例化LossHistory类 保存每次训练的loss
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)#实例化LossHistory类

    
    if True:
        UnFreeze_flag = False
        batch_size = Unfreeze_batch_size
        #   判断当前batch_size，自适应调整学习率
        # 9. 调整学习率
        nbs             = 64
        lr_limit_max    = 5e-2
        lr_limit_min    = 5e-5
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max) #INIT_LR = 2e-3 batch_size = 8 
        #Init_lr_fit的值计算为：min(max(8/64*2e-3, 5e-5), 5e-2) = 2e-3 *（1/8）=0.00025
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2) #MIN_LR = 2e-5 batch_size = 8
        #Min_lr_fit的值计算为 min(max(8/64*2e-5, 5e-5*1e-2), 5e-2) =  2.5e-6
        #10.SGD优化器
        optimizer = optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)#选择优化器
        #11.获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch) #   获得学习率下降的公式
        

        epoch_step      = num_train // batch_size#训练集的步长
        epoch_step_val  = num_val // batch_size#验证集的步长

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        #12.使用SSDDataset类实例化训练集和验证集的数据加载器
        train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)#训练集的数据加载器
        val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)#验证集的数据加载器
        
     
        train_sampler   = None
        val_sampler     = None
        shuffle         = True
        # 训练集和验证集的数据加载器
        #13.使用DataLoader类实例化训练集和验证集的数据加载器
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=ssd_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=ssd_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        
        #14. 开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            batch_size = Unfreeze_batch_size
            
            #   判断当前batch_size，自适应调整学习率
            nbs             = 64
            lr_limit_max    =  5e-2
            lr_limit_min    =  5e-5
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            #   获得学习率下降的公式
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
            if backbone == "vgg":
                for param in model.vgg[:28].parameters():
                    param.requires_grad = True
            
                        
            gen         = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=ssd_dataset_collate, sampler=train_sampler, 
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))#训练集的数据加载器
            gen_val     = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=ssd_dataset_collate, sampler=val_sampler, 
                                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))#验证集的数据加载器


            #15.根据当前的epoch设置优化器的学习率
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            #16.训练模型
            fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
  


