import math
from functools import partial

import torch
import torch.nn as nn


class MultiboxLoss(nn.Module):

    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        #当前的类别数量
        self.num_classes = num_classes
        #alpha是用来控制置信度损失的比重 loss = L_conf + alpha * L_loc 根据论文指出根据交叉验证计算出来1.0比较合适
        self.alpha = alpha
        #负正样本比例 3:1
        self.neg_pos_ratio = neg_pos_ratio
        #背景类别的id
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        #负样本的数量
        self.background_label_id = background_label_id
        #每张图片的负样本数量 100
        
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]
        # negatives_for_hard 的shape是torch.Size([1])，所以要加[0]变成torch.Size([])
        

    def _l1_smooth_loss(self, y_true, y_pred):
        # 计算预测值和真实值的差值
        abs_loss = torch.abs(y_true - y_pred)
        # 绝对值 小于1的部分使用0.5x^2作为loss，大于1的部分使用|x|-0.5作为loss
        sq_loss = 0.5 * (y_true - y_pred)**2
        
        #使用where函数，如果abs_loss小于1，那么使用sq_loss，否则使用abs_loss-0.5作为loss
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        # 将每个框的loss相加
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        # 计算softmax的loss
        # y_pred的值域是[0,1]，但是由于log函数的特性，我们需要将y_pred的值域变成[1e-7,1]
        # clamp函数的作用就是将y_pred的值域限制在[1e-7,1]
        # clamp参数min表示最小值，max表示最大值 限制最小值为1e-7
        y_pred = torch.clamp(y_pred, min = 1e-7)
        # torch.log(y_pred)表示求y_pred的自然对数
        # 对所有的类别求和
        # y_true * torch.log(y_pred)表示将y_true中为1的部分保留下来，为0的部分变成0
        # torch.sum(y_true * torch.log(y_pred),axis=-1)表示对最后一维求和
       
        # y_true=0表示背景类别，y_true=1表示正样本
        softmax_loss = -torch.sum(y_true * torch.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        # --------------------------------------------- #
        #   y_true batch_size, 8732, 4 + self.num_classes + 1
        #   y_pred batch_size, 8732, 4 + self.num_classes
        # --------------------------------------------- #
        
        
        num_boxes       = y_true.size()[1]
        # numboxes是8732
        
        y_pred          = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim = -1)
        # y_pred目前的shape为torch.Size([2, 8732, 25])
        #y_pred[0]是预测的框的位置，y_pred[1]是预测的框的类别
        #y_pred[0] shape是torch.Size([2, 8732, 4])，y_pred[1] shape是torch.Size([2, 8732, 21])
        '''
        y_pred[0] 是一个张量，假设它包含了模型的某些预测值。它是拼接操作的第一个输入张量。

        nn.Softmax(-1)(y_pred[1]) 是一个张量，它首先应用了 softmax 操作到 y_pred[1] 上，
        然后生成一个包含概率分布的张量。这是拼接操作的第二个输入张量。

        dim=-1 表示在最后一个维度上进行拼接，也就是列维度。这是拼接操作的维度参数。
        '''
        # --------------------------------------------- #
        #   分类的loss
        #   batch_size,8732,21 -> batch_size,8732
        # --------------------------------------------- #
        # 1.计算分类的loss y_true[:, :, 4:-1]表示取y_true的第4个维度到倒数第二个维度 5-24 y_pred[:, :, 4:]表示取y_pred的第4个维度到最后一个维度 4-24
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])
        
        # --------------------------------------------- #
        #   框的位置的loss
        #   batch_size,8732,4 -> batch_size,8732
        # --------------------------------------------- #
        # 2.计算框的位置的loss y_true[:, :, :4]表示取y_true的第一个维度到第四个维度 0-3 y_pred[:, :, :4]表示取y_pred的第一个维度到第四个维度 0-3
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # --------------------------------------------- #
        #   获取所有的正标签的loss
        # --------------------------------------------- #
        # 3.获取所有的正标签loc loss y_true[:, :, -1]表示取y_true的最后一个维度 -1 
        # y_true[:, :, -1]表示是否包含物体
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],
                                     axis=1)
        # 4.获取所有的正标签分类loss y_true[:, :, -1]表示取y_true的最后一个维度 -1
        # y_true[:, :, -1]表示是否包含物体
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],
                                      axis=1)

        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   num_pos     [batch_size,]
        # --------------------------------------------- #
        # 5.每一张图的正样本的个数 即每个batch中每张图的正样本的数量
        num_pos = torch.sum(y_true[:, :, -1], axis=-1)
         
        # --------------------------------------------- #
        #   每一张图的负样本的个数
        #   num_neg     [batch_size,]
        # --------------------------------------------- #

        # 6.每一张图的负样本的个数  取剩余的框的数量减去正样本的数量和neg_pos_ratio（3）倍数的正样本数量的最小值
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)

        # 7.pos_num_neg_mask shape是torch.Size([2, 8732]) 代表每张图的负样本的数量是否大于0
        pos_num_neg_mask = num_neg > 0 
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        
        has_min = torch.sum(pos_num_neg_mask)

        # --------------------------------------------- #
        #   从这里往后，与视频中看到的代码有些许不同。
        #   由于以前的负样本选取方式存在一些问题，
        #   我对该部分代码进行重构。
        #   求整个batch应该的负样本数量总和
        # --------------------------------------------- #

        # 8.求整个batch应该的负样本数量总和 如果所有的图，正样本的数量均为0 那么则默认选取100个先验框作为负样本
        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard
        
        # --------------------------------------------- #
        #   对预测结果进行判断，如果该先验框没有包含物体
        #   那么它的不属于背景的预测概率过大的话
        #   就是难分类样本
        # --------------------------------------------- #
        confs_start = 4 + self.background_label_id + 1
        #background_label_id=0
        #confs_start=5 代表第5个维度开始是不属于背景的预测概率

        confs_end   = confs_start + self.num_classes - 1
        #confs_end=25 代表第25个维度结束是不属于背景的预测概率
        
        # --------------------------------------------- #
        #   batch_size,8732
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)
        #y_pred[:, :, confs_start:confs_end]表示取y_pred的第5个维度到第25个维度 4-24
        #torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)表示对第三个维度求和
        #max_confs shape是torch.Size([2, 8732]) 代表每张图的不属于背景的预测概率求和
        # max_confs大的话，代表越难分类 anchors就是难分类样本
        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        
        # 9.只有没有包含物体的先验框才得到保留 这就是实际上不包含物体的anchors的所有种类预测概率求和
        max_confs   = (max_confs * (1 - y_true[:, :, -1])).view([-1])
        #max_confs的shape是torch.Size([17464]) 代表所有的anchors的所有种类预测概率求和
        #y_true[:, :, -1]表示取y_true的最后一个维度 -1
        
        _, indices  = torch.topk(max_confs, k = int(num_neg_batch.cpu().numpy().tolist()))
        #indices shape是torch.Size([100]) 代表选取的100个难分类样本的索引
        #indices是max_confs中最大的100个值的索引

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)
        #neg_conf_loss shape是torch.Size([100]) 代表选取的100个难分类样本的分类loss
        #conf_loss.view([-1])表示将conf_loss的shape变成torch.Size([17464]) 代表所有的anchors的分类loss
        #indices表示选取的100个难分类样本的索引
        #torch.gather(conf_loss.view([-1]), 0, indices)表示选取的100个难分类样本的分类loss
        
        # 进行归一化
        num_pos     = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        #num_pos shape是torch.Size([2]) 代表每张图的正样本的数量
        #torch.ones_like(num_pos)表示生成一个和num_pos相同shape的全1张量
        #torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))表示如果num_pos不等于0，那么就取num_pos，否则就取torch.ones_like(num_pos) 全一张量
        

        total_loss  = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        #total_loss shape是torch.Size([]) 代表总的loss
        #torch.sum(pos_conf_loss)表示所有的正样本的分类loss相加
        # 正样本的分类loss + 负样本的分类loss + alpha * 正样本的框的位置loss （负样本取的数量为neg_pos_ratio倍的正样本）

        total_loss  = total_loss / torch.sum(num_pos)#torch.sum(num_pos)表示所有的正样本的数量相加
        return total_loss


# get_lr_scheduler函数用来获取学习率调整函数 余弦退火
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    return func



#设置优化器的学习率
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    #根据epoch计算当前的学习率
    lr = lr_scheduler_func(epoch)
    #将学习率设置到optimizer中
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    