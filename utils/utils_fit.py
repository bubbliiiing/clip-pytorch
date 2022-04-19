import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss              = 0
    total_image_accuracy    = 0
    total_text_accuracy     = 0

    val_total_loss      = 0
    val_image_accuracy  = 0
    val_text_accuracy   = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, captions = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            logits_per_image, logits_per_text   = model_train(images, captions)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text
            
            loss.backward()
            optimizer.step()
            
        else:
            from torch.cuda.amp import autocast
            with autocast():
                logits_per_image, logits_per_text   = model_train(images, captions)
                loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss                                = loss_logits_per_image + loss_logits_per_text
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        total_loss += loss.item()
        with torch.no_grad():
            image_accuracy      = torch.mean((torch.argmax(logits_per_image.softmax(dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            text_accuracy       = torch.mean((torch.argmax(logits_per_text.softmax(dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            total_image_accuracy    += image_accuracy.item()
            total_text_accuracy     += text_accuracy.item()
            
        pbar.set_postfix(**{'total_loss'            : total_loss / (iteration + 1), 
                            'total_image_accuracy'  : total_image_accuracy / (iteration + 1), 
                            'total_text_accuracy'   : total_text_accuracy / (iteration + 1),
                            'lr'                    : get_lr(optimizer)})
        pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, captions = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

            optimizer.zero_grad()
    
            logits_per_image, logits_per_text   = model_train(images, captions)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text
            
            val_total_loss += loss.item()
            
            with torch.no_grad():
                image_accuracy      = torch.mean((torch.argmax(logits_per_image.softmax(dim=-1), dim=-1) == labels).type(torch.FloatTensor))
                text_accuracy       = torch.mean((torch.argmax(logits_per_text.softmax(dim=-1), dim=-1) == labels).type(torch.FloatTensor))
                val_image_accuracy    += image_accuracy.item()
                val_text_accuracy     += text_accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'              : val_total_loss / (iteration + 1), 
                                'val_image_accuracy'    : val_image_accuracy / (iteration + 1), 
                                'val_text_accuracy'     : val_text_accuracy / (iteration + 1),
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch, total_loss / epoch_step, val_total_loss / epoch_step_val)
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_total_loss / epoch_step_val))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch+1), total_loss / epoch_step, val_total_loss / epoch_step_val)))
