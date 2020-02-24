from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import logging
import os
#加载自定义文件
import models
from attack.attack_pp import MPGD
from utils import init_prog, process_img, tensor2img,\
        calc_mse, add_arguments, print_arguments, img2tensor,\
        input_diversity, get_bbox, area_rank


import cv2
verbose = True

#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   121,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "./input_image/",     "Input directory with images")
add_arg('output',           str,   "./output_image/",    "Output directory with images")
add_arg('step_size',        float,  0.1,                 "step size")
add_arg('norm_regulizer',   float,  0.005,               "norm regulizer")
add_arg('clip_eps',         float,  0.3,                 "Lp norm limitation")
add_arg('verbose',          int,    0,                   "verbose")
add_arg('attack_iters',     int,    100,                 "attack iterations")
add_arg('diversity_iter',   int,    1,                  "diversity iterations")
add_arg('attack_confidence',float,  0.8,                 "attack confidence")
add_arg('sparse_percentage', float, -1,                  "sparse gradient")
add_arg('momentum_decay',   float,  0.9,                  "momentum_decay")
add_arg('lr_decay',         float,  1.0,                  "lr_decay")
add_arg('start', int, 0, "start")
add_arg('end', int, 120, "end")

parser.add_argument("--gradient_sign",dest='gradient_sign',action="store_true")
parser.add_argument("--istarget",dest='istarget',action="store_true")
parser.add_argument('--models', nargs='+', type=str, default=["ResNeXt50_32x4d"], help="model name list")
parser.add_argument('--weights', nargs='+', type=float, default=[1.0], help="model name list")
#--models MobileNetV2_x2_0  ResNeXt50_32x4d GoogleNet

args = parser.parse_args()
print_arguments(args)
logging.basicConfig(filename=os.path.join(args.output,'attack.log'), level=logging.INFO)
def save_adv_image(img, output_path):
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return

######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
init_dir = '../INIT/'
output_dir = args.output
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
modelnames = args.models
weights = args.weights
#weights = np.ones(len(modelnames),dtype=np.float32) / len(modelnames)
pretrained_model = "../checkpoint/"
val_list = 'val_list.txt'
use_gpu=True
diversity_iter = args.diversity_iter
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

SPARSE_PER = 95
######Attack graph
adv_program=fluid.Program()
#完成初始化
with fluid.program_guard(adv_program):
    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    noise_layer = fluid.layers.data(name='noise', shape=image_shape, dtype='float32')
    reg_const = fluid.layers.data(name='reg', shape=[1], dtype='float32',append_batch_size=False)
    zero_input = fluid.layers.zeros_like(noise_layer)
    #设置为可以计算梯度
    noise_layer.stop_gradient=False
    raw_input = input_layer + noise_layer
    input_argued = input_diversity(raw_input, diversity_iter>1)
    out_logits = 0
    # model definition
    for i in range(len(modelnames)):
        model = models.__dict__[modelnames[i]]()
        out_logits += model.net(input=input_argued, class_dim=class_dim) * weights[i]
    out = fluid.layers.softmax(out_logits)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #记载模型参数
    fluid.io.load_persistables(exe, pretrained_model)
#设置adv_program的BN层状态
init_prog(adv_program)
#创建测试用评估模式
eval_program = adv_program.clone(for_test=True)

#定义梯度
with fluid.program_guard(adv_program):
    label = fluid.layers.data(name="label", shape=[1] ,dtype='int64')
    loss_cls = fluid.layers.cross_entropy(input=out, label=label)
    loss_norm = fluid.layers.mse_loss(noise_layer,zero_input)
    loss = loss_cls - loss_norm * reg_const
    gradients = fluid.backward.gradients(targets=loss, inputs=[noise_layer])[0]

######Inference
def inference(img):
    fetch_list = [out.name]

    result = exe.run(eval_program,
                     fetch_list=fetch_list,
                     feed={'image':img, 'noise':np.zeros_like(img)})
    result = result[0][0]
    pred_label = np.argmax(result)
    pred_score = result[pred_label].copy()
    return pred_label, pred_score

####### Main #######
def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def get_init_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile][:20]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def attack_by_MPGD(src_img, src_label, bboxes, sparse_per,init=None):
    verbose = args.verbose
    step_size = args.step_size
    epsilon = args.clip_eps
    attack_iters = args.attack_iters
    attack_confidence = args.attack_confidence
    sparse_percentage = args.sparse_percentage
    #sparse_percentage = sparse_per
    momentum_decay = args.momentum_decay
    lr_decay = args.lr_decay
    init = init
    adv = MPGD(adv_program, gradients, src_img, src_label, input_layer, noise_layer,
                out, loss, loss_cls, bboxes,
    step_size=step_size, epsilon=epsilon, iteration=attack_iters, lr_decay=lr_decay,use_gpu=use_gpu,
    confidence=attack_confidence, verbose=verbose, init=init, decay_factor=momentum_decay, 
    diversity_iter=diversity_iter,sparse_percentage = sparse_percentage, gradient_sign = args.gradient_sign,
    norm_regulizer = args.norm_regulizer)
    adv_img=tensor2img(adv)
    pred_label, pred_score = inference(adv)
    return adv_img, pred_label

def gen_adv():
    mse = 0
    adv_acc = 0
    original_files = get_original_file(input_dir + val_list)
    #init_files = get_init_file(init_dir+'init_list.txt')
    for idx, (filename, label) in enumerate(original_files[args.start:args.end]):
        img_path = input_dir + filename
        #init_path = init_dir +  init_files[idx][0] + '.jpg'
        image_name, image_ext = filename.split('.')
        if image_name in area_rank[:40]:
            SPARSE_PER = 99
        if image_name in area_rank[40:80]:
            SPARSE_PER = 97
        if image_name in area_rank[80:]:
            SPARSE_PER = 95
        #bboxes = get_bbox('mask/'+image_name+'.xml')
        bboxes = None
        if verbose:
            print("Image: {0} ".format(img_path))
        img=process_img(img_path)
        #init = process_img(init_path) * 0.01
        init = None
        adv_img, adv_label = attack_by_MPGD(img, label, bboxes, SPARSE_PER, init)
        save_adv_image(adv_img, output_dir+image_name+'.png')
        org_img = tensor2img(img)
        score = calc_mse(org_img, adv_img)

        mse += score if label != adv_label else 128
        adv_acc += 1 if label == adv_label else 0
        if label == adv_label:
            print("model: ",i,"\timage: ",filename,label)
    print("ADV {} files, AVG MSE: {}, ADV_ACC: {} ".\
            format(len(original_files), mse/len(original_files),adv_acc))

def main():
    print("="*100)
    gen_adv()
if __name__ == '__main__':
    main()