#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import math
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
from utils import *
import six
from tqdm import tqdm
import logging

from scipy.ndimage import gaussian_filter

def l2norm(adv_img): # 使用L2正则化 返回图片与零图片的差值
    org_img = np.zeros_like(adv_img) # 返回一个与给定类型,形状相似的,全零的矩阵
    diff = org_img.reshape((-1, 3)) - adv_img.reshape((-1, 3))# 求的输入图片与全零矩阵的差
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1))) 
    return distance

#实现linf约束 输入格式都是tensor 返回也是tensor [1,3,224,224]
def linf_img_tenosr(o,adv,epsilon=16.0/256):
    
    o_img=tensor2img(o).astype(np.float32) # 将tensor转换为float32格式的图片
    adv_img=tensor2img(adv).astype(np.float32)
    
    clip_max=np.clip(o_img*(1.0+epsilon),0,255) # 使用np clip 将图片 转换到0～255之间,得到该图片最小、最大矩阵
    clip_min=np.clip(o_img*(1.0-epsilon),0,255)
    
    adv_img=np.clip(adv_img,clip_min,clip_max) # 将adv图片按照最大最小矩阵转换。
    
    adv_img=img2tensor(adv_img) # 将adv转换为tensor 输出
    
    return adv_img


def MPGD(adv_program, gradients, o_img, o_label, input_layer, noise_layer, output_layer, loss_layer, loss_cls,
        bboxs, step_size=2.0/256, epsilon=16.0/256, iteration=20,lr_decay=1.0,use_gpu=True,
        confidence=0.8, verbose=True, init=None, decay_factor=0.8, diversity_iter=1, 
        sparse_percentage=-1, gradient_sign=False, norm_regulizer=0.0):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

        #无定向攻击 target_label的值自动设置为原标签的值
    if verbose:
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
    logging.info("Non-Targeted attack target_label=o_label={}".format(o_label))
    target_label=o_label

               
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    noise = np.zeros_like(o_img) 
    if not init is None:
        noise = init
    momentum = 0
    history=0
    history_bst = -1
    for i in range(iteration):
        norm_reg = np.zeros((1),dtype=np.float32) if i < -1 else np.array([norm_regulizer],dtype=np.float32)
        #计算梯度
        g_sum = 0
        for j in range(diversity_iter):
            g,pred,loss, l_cls = exe.run(adv_program,
                            fetch_list=[gradients, output_layer, loss_layer, loss_cls],
                            feed={input_layer.name : o_img, noise_layer.name : noise, 'label': target_label, 'reg':norm_reg}
                    )
            g_sum += g
        g = g_sum / diversity_iter
        # for j in range(3):
        #     g[0,j,:,:] = gaussian_filter(g[0,j,:,:],0.8)
        # if bboxs is not None:
        #     mask = np.zeros_like(g)
        #     for b in bboxs:
        #         mask[:,:,b[0]:b[1],b[2]:b[3]] = 1
        #     g = g * mask
        pred = pred[0]
        loss = loss[0]
        l_cls = l_cls[0][0]
        adv_label = np.argsort(pred)[::-1][:1][0]
        adv_score = pred[adv_label]
        o_score = pred[o_label]
        if l_cls > history_bst:
            history_bst = l_cls
            history = 0
        else:
            history +=1
        if history > 10:
            return o_img + noise
        if verbose:
            print("Iter: %d  Loss_cls: %.7f, Loss: %.7f, Adv Confidence: %.3f, Adv Label: %d  Adv MSE: %.7f"%(i+1, l_cls, loss, adv_score-o_score, adv_label, calc_mse(tensor2img(o_img + noise),tensor2img(o_img))))
        logging.info("Iter: %d  Loss_cls: %.3f, Loss: %.3f, Adv Confidence: %.3f, Adv Label: %d  Adv MSE: %.3f"%(i+1, l_cls, loss, adv_score-o_score, adv_label, calc_mse(tensor2img(o_img + noise),tensor2img(o_img))))
        if adv_score-o_score > confidence:
            return o_img + noise
        if sparse_percentage > 0:
            sparse_thres = np.percentile(np.abs(g),sparse_percentage)
            g = np.where(g >= sparse_thres, g, 0) + np.where(g < -sparse_thres, g, 0)
        if gradient_sign:
            g = np.sign(g)
        else:
            g = g / (np.mean(np.abs(g),(2,3),keepdims=True))
        velocity = g
        #momentum = decay_factor * momentum + velocity
        momentum = decay_factor * momentum + (1-decay_factor) * velocity
        #momentum = momentum / (np.mean(np.abs(momentum),(2,3),keepdims=True))
        noise = noise + momentum * step_size
        #实施linf约束
        if step_size > 0.0005:
            step_size *=lr_decay
        noise = linf_img_tenosr(o_img, o_img+noise, epsilon) - o_img 
        # bound = min(0.08,l2norm(noise))
        # noise = noise * bound / l2norm(noise)

    return o_img + noise

