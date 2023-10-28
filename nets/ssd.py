import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from nets.vgg import vgg as add_vgg


# 使用的特征图 Conv4_3 FC7(前vgg) Conv6_2 Conv7_2 Conv8_2 Conv9_2(额外添加的)
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
   
    # Block 6
    # 19,19,1024 -> 19,19,256 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 10,10,128 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 5,5,128 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    
    # Block 9
    # 3,3,256 -> 3,3,128 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)

class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(SSD300, self).__init__()
        self.num_classes    = num_classes
        if backbone_name    == "vgg":
            self.vgg        = add_vgg(pretrained)
            self.extras     = add_extras(1024, backbone_name)
            self.L2Norm     = L2Norm(512, 20)
            mbox            = [4, 6, 6, 6, 4, 4]
            loc_layers      = []
            conf_layers     = []
            #Conv4_3 在第21层 
            #FC6 在倒数第二层 最后一层是Relu
            backbone_source = [21, -2]
            #---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            #---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            #-------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            #-------------------------------------------------------------#  
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
 

            #defaultbox 数量 每个特征图对应的每个特征点拥有的default box 个数
            mbox            = [4, 6, 6, 6, 4, 4]
            
            loc_layers      = []
            conf_layers     = []
            out_channels    = [512, 1024]
            #---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第layer3层和layer4层可以用来进行回归预测和分类预测。
            #---------------------------------------------------#
            for k, v in enumerate(out_channels):
                loc_layers  += [nn.Conv2d(out_channels[k], mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(out_channels[k], mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            #-------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            #-------------------------------------------------------------#  
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
        else:
            raise ValueError("The backbone_name is not support")
        
        #loc_layers/conf_layers 参数为（通道数，每个特征点对应的defaultbox 数量，卷积核的大小，填充）
        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        #主干名为backbone 
        self.backbone_name  = backbone_name
    #torch.Size([2, 3, 300, 300])
    #x (batch_size, input_channels, input_height, input_width)    
    def forward(self, x):
        #---------------------------#
        #   x是300,300,3
        #---------------------------#
        sources = list()
        loc     = list()
        conf    = list()

        #---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x= self.vgg[k](x)

        #---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        #---------------------------#
        s = self.L2Norm(x)
        sources.append(s)
        #soures 中单个元素值为 （输入通道数，输出通道数，卷积核大小，padding填充）
        #---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)

        sources.append(x)
        #-------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#      
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg" :
                if k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)

        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------#  
        #  l 是 loc 回归层 c 是分类层    
        # l/c（通道数，每个特征点对应的defaultbox 数量，卷积核的大小，填充）
        # x 是source 获得6个特征层 
        # loc[0] (16X38X38X16)
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) 
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        #loc 现在的值(batch_size, height, width,channels) 
        # 其中channel 用作回归预测  大小为 defaultbox * 4(4是四个坐标)
        # conf 其中channel 用作回归预测  大小为 defaultbox * classnum(类的数量)
        '''
        将维度从 (batch_size, channels, height, width) 重新排列为 (batch_size, height, width, channels)。
        这通常是因为在目标检测的常见表示中，最后一个维度通常表示不同的类别的预测。
        ''' 
        
        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#
  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output
