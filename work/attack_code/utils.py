#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import sys
import math
import numpy as np
import argparse
import functools
import distutils.util
import six

from PIL import Image, ImageOps
#绘图函数
import matplotlib
#服务器环境设置
import matplotlib.pyplot as plt

import paddle.fluid as fluid
import random
random.seed(123)
import xml.etree.ElementTree as ET


#去除batch_norm的影响
def init_prog(prog):
    for op in prog.block(0).ops:
        #print("op type is {}".format(op.type))
        if op.type in ["batch_norm"]:
            # 兼容旧版本 paddle
            if hasattr(op, 'set_attr'):
                op.set_attr('is_test', False)
                op.set_attr('use_global_stats', True)
            else:
                op._set_attr('is_test', False)
                op._set_attr('use_global_stats', True)
                op.desc.check_attrs()

def img2tensor(img,image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.resize(img,(image_shape[1],image_shape[2]))

    #RGB img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
     
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

def process_img(img_path="",image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.imread(img_path)
    img = cv2.resize(img,(image_shape[1],image_shape[2]))
    #img = cv2.resize(img,(256,256))
    #img = crop_image(img, image_shape[1], True)
    
    #RBG img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    #img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def tensor2img(tensor):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    
    img=tensor.copy()
      
    img *= img_std
    img += img_mean
    
    img = np.round(img*255) 
    img = np.clip(img,0,255)

    img=img[0].astype(np.uint8)
        
    img = img.transpose(1, 2, 0)
    img = img[:, :, ::-1]
    
    return img

def save_adv_image(img, output_path):
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return

def calc_mse(org_img, adv_img):
    diff = org_img.astype(np.float).reshape((-1, 3)) - adv_img.astype(np.float).reshape((-1, 3))
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
    return distance

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


# def cast_int(x):
#     x = fluid.layers.round(x)
#     x = fluid.layers.cast(x,dtype=np.int32)
#     return x
# def input_diversity(input_layer):
#     rnd_gauss = fluid.layers.uniform_random(shape=[1], dtype='float32', min=0.0, max=0.5)
#     rnd_zoom = fluid.layers.uniform_random(shape=[2], dtype='float32', min=224.0, max=256.0)
#     rnd_zoom = cast_int(rnd_zoom)
#     gauss_input = input_layer +  fluid.layers.elementwise_mul(fluid.layers.gaussian_random_batch_size_like(input_layer,shape=[-1,3,224,224]),rnd_gauss)
#     rescaled = fluid.layers.image_resize(gauss_input, out_shape=rnd_zoom)
#     input_argued = fluid.layers.random_crop(rescaled, shape=[3, 224, 224])
#     return input_argued

def batchimg2tensor(img):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 

    #RGB img [N, 224,224,3]->[N, 3,224,224]
    img = img[:, :, :, ::-1].astype('float32').transpose((0, 3, 1, 2)) / 255 
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    return img


def batchtensor2img(tensor):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    
    img=tensor.copy()
      
    img *= img_std
    img += img_mean
    
    img = np.round(img*255) 
    img = np.clip(img,0,255)

    img=img.astype(np.uint8)
        
    img = img.transpose(0, 2, 3, 1)
    img = img[:, :, :, ::-1]
    return img
def input_diversity(input_layer,diversity):
    if not diversity:
        return input_layer
    rnd_gauss = fluid.layers.uniform_random(shape=[1], dtype='float32', min=0.0, max=0.5, seed=123)
    # rnd_zoom = fluid.layers.uniform_random(shape=[2], dtype='float32', min=224.0, max=256.0)
    # rnd_zoom = cast_int(rnd_zoom)
    gauss_input = input_layer +  fluid.layers.elementwise_mul(fluid.layers.gaussian_random_batch_size_like(input_layer,shape=[-1,3,224,224],std=1.0, seed = 123),rnd_gauss)
    # rescaled = fluid.layers.image_resize(gauss_input, out_shape=rnd_zoom,resample="NEAREST")
    # input_argued = fluid.layers.random_crop(rescaled, shape=[3, 224, 224])
    return gauss_input
    
def get_bbox(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size=root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    scale_x = 224 / width
    scale_y = 224 / height
    bboxes = []
    for obj in root.iter('object'):
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)
        xmin *= scale_x
        xmax *= scale_x
        ymin *= scale_y
        ymax *= scale_y
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([ymin,ymax,xmin,xmax])
    return bboxes

area_rank = ['n02111277_10237', 'n02107683_1003', 'n02097130_1193', 'n02096177_10031', 'n02098286_1009', 'n02099712_1150', 'n02094114_1173', 'n02116738_10024', 'n02097658_1018', 'n02110958_10378', 'n02095570_1031', 'n02090622_10343', 'n02113712_10525', 'n02100583_10249', 'n02105505_1018', 'n02091032_10079', 'n02106166_1205', 'n02088364_10108', 'n02099429_1039', 'n02091831_10576', 'n02096437_1055', 'n02105162_10076', 'n02087046_1206', 'n02110063_11105', 'n02092339_1100', 'n02098105_1078', 'n02113023_1136', 'n02106030_11148', 'n02093859_1003', 'n02090721_1292', 'n02115641_10261', 'n02110185_10116', 'n02109961_11224', 'n02100735_10064', 'n02087394_11337', 'n02097474_1070', 'n02088238_10013', 'n02097047_1412', 'n02093991_1026', 'n02105855_10095', 'n02112137_1005', 'n02088094_1003', 'n02113624_1461', 'n02093754_1062', 'n02105641_10051', 'n02100236_1244', 'n02088632_101', 'n02112350_10079', 'n02108551_1025', 'n02085936_10130', 'n02108915_10564', 'n02111500_1048', 'n02090379_1272', 'n02109525_10032', 'n02102177_1160', 'n02108000_1087', 'n02086910_1048', 'n02094433_10126', 'n02113978_1034', 'n02110627_10147', 'n02093428_10947', 'n02096585_10604', 'n02104365_10071', 'n02102973_1037', 'n02109047_10160', 'n02091244_1000', 'n02111889_10059', 'n02089078_1064', 'n02085620_10074', 'n02102480_101', 'n02089867_1029', 'n02096051_1110', 'n02107312_105', 'n02107574_1026', 'n02105251_1588', 'n02099601_100', 'n02108089_1104', 'n02088466_10083', 'n02110806_1214', 'n02086646_1002', 'n02108422_1096', 'n02094258_1004', 'n02095889_1003', 'n02101388_10017', 'n02092002_10699', 'n02097298_10676', 'n02102318_10000', 'n02091467_1110', 'n02086079_10600', 'n02101556_1116', 'n02112018_10158', 'n02107142_10952', 'n02105056_1165', 'n02107908_1030', 'n02115913_1010', 'n02099849_1068', 'n02113186_1030', 'n02093256_11023', 'n02106550_10048', 'n02111129_1111', 'n02113799_1155', 'n02096294_1111', 'n02095314_1033', 'n02086240_1059', 'n02085782_1039', 'n02089973_1066', 'n02098413_11385', 'n02105412_1159', 'n02106382_1005', 'n02099267_1018', 'n02091134_10107', 'n02102040_1055', 'n02091635_1319', 'n02100877_1062', 'n02097209_1038', 'n02112706_105', 'n02093647_1037', 'n02104029_1075', 'n02106662_10122', 'n02101006_135']