import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

#这个函数用于训练一个epoch
#其中包括了训练和验证
#训练时需要计算loss，验证时不需要计算loss
#训练时需要计算loss的原因是为了更新权值
#验证时不需要计算loss的原因是为了加快验证速度
#这个函数的输入包括：
#   model_train：训练模型
#   model：验证模型
#   ssd_loss：损失函数
#   loss_history：保存训练和验证的loss
#   eval_callback：验证回调函数
#   optimizer：优化器
#   epoch：当前训练的epoch
#   epoch_step：训练的步数
#   epoch_step_val：验证的步数
#   gen：训练数据集
#   gen_val：验证数据集
#   Epoch：总的训练周期
#   cuda：是否使用cuda
#   fp16：是否使用半精度训练
#   scaler：半精度训练的缩放器
#   save_period：保存模型的周期
#   save_dir：保存模型的路径
#   local_rank：当前进程的编号
def fit_one_epoch(model_train, model, ssd_loss,loss_history , optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss  = 0
    val_loss    = 0 

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        '''
        total：指定了总的步数（总进度的最大值），通常用来表示在整个进度条中的总步骤数。在这里，epoch_step变量的值被用作总步数。

        desc：进度条的描述文本，通常用来说明进度条所代表的操作或任务。在这里，描述文本是一个字符串，它包括了当前的训练周期（epoch）和总周期数（Epoch）。

        postfix：这是一个字典（dictionary），其中包含了要在进度条右侧显示的额外信息。这可以包括一些有关进度的额外信息，如损失值、准确度等。

        mininterval：指定了更新进度条的最小时间间隔，以避免过于频繁的更新。在这里，最小时间间隔被设置为0.3秒。
        '''
    #local_rank是指当前进程的编号，如果是单GPU训练，那么local_rank为0

    model_train.train()
    # train的函数定义为model.train()，这个函数的作用是启用 BatchNormalization 和 Dropout，将 BatchNormalization 设置为 True，Dropout 设置为 True。
    # 下面一个循环为一个步长
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
            #如果在cuda上面运行，那么将数据转换到cuda上面
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            out = model_train(images)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   计算损失
            #----------------------#
            loss = ssd_loss.forward(targets, out)
            #target是真实框，out是预测框，计算损失
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                out = model_train(images)
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   计算损失
                #----------------------#
                loss = ssd_loss.forward(targets, out)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            #解释一下以上代码
        '''
        'total_loss': total_loss / (iteration + 1)：这个键值对表示显示在进度条右侧的额外信息，其中 'total_loss' 是键
        ，total_loss / (iteration + 1) 是值。这里，total_loss 表示累积的损失值，它被除以 (iteration + 1)，以计算平均损失值。
        这个平均损失值将在进度条中显示，用来表示损失值的趋势。
        'lr': get_lr(optimizer)：这个键值对表示学习率信息，其中 'lr' 是键，get_lr(optimizer) 是值。get_lr(optimizer) 是一个函数，用来获取当前优化器（optimizer）的学习率（learning rate）。
        学习率通常也会显示在进度条中，以帮助用户了解训练中学习率的变化情况。
        pbar.update(1)：这行代码用于每次迭代结束后更新进度条，每次调用 update(1) 表示前进一个步骤。这有助于在进度条中显示训练进度的变化。
        '''
                
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        #开始验证
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()#关闭dropout和BN eval是评估模式，这个函数的作用是不启用 BatchNormalization 和 Dropout，将 BatchNormalization 设置为 False，Dropout 设置为 False。
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            out     = model_train(images)
            optimizer.zero_grad()
            loss    = ssd_loss.forward(targets, out)
            val_loss += loss.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'      : val_loss / (iteration + 1), 
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        #loss_history加上当前的训练和验证的loss
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)



        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        #save_period是保存模型的周期
        #当训练次数等于save_period或者等于Epoch时，就保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            #保存模型 它的名称为ep%03d-loss%.3f-val_loss%.3f.pth 训练loss和验证loss
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            #如果验证的loss小于最小的loss，那么就保存模型（best_epoch_weights.pth）
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))