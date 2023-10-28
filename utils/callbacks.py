import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import BBoxUtility

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        os.makedirs(self.log_dir)
      

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):#如果路径不存在
            os.makedirs(self.log_dir)#创建路径
        self.losses.append(loss)#将loss添加到列表中
        self.val_loss.append(val_loss)#将val_loss添加到列表中
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:#打开文件
            f.write(str(loss))#将loss写入文件中
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:#打开文件
            f.write(str(val_loss))#将val_loss写入文件中
            f.write("\n")

   
