import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union

import matplotlib.pyplot as plt
import time
import cv2
from collections import OrderedDict

def visualize_segmentation_with_mask(folder):
    _train = True #training mode
    _exp = 'bn_lr7e-3' #name of experiment
    _backbone = 'resnet101' 
    _dataset = 'pascal' #model: pascal or cityscapes
    _groups = None #num of groups for group normalization
    _epochs = 50 #num of training epochs
    _batch_size = 8 #batch size
    _base_lr = 0.007 #base learning rate
    _last_mult = 1.0 #learning rate multiplier for last layers
    _scratch = False #train from scratch
    _freeze_bn = False #freeze batch normalization parameters
    _weight_std = False #weight standardization
    _beta = False #resnet101 beta
    _crop_size = 513 #image crop size
    _resume = None #path to checkpoint to resume from
    _workers = 4 #number of data loading workers
    if not os.path.exists(folder):
        print('file \"{}\" doesn\'t exist.'.format(folder))
        return
    torch.backends.cudnn.benchmark = True
    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(_backbone, _dataset, _exp)
    dataset = VOCSegmentation('data/VOCdevkit', train=_train, crop_size=_crop_size)
    model = getattr(deeplab, 'resnet101')(
                pretrained=(not _scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=_groups,
                weight_std=_weight_std,
                beta=_beta)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname % _epochs)
    state_dict =checkpoint['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
    ###load vedio
    pics = os.listdir(folder)
    pics.sort()
    for i in range(len(pics)):
        ###segmentation
        img_path = './'+folder+'/'+pics[i]
        test_img = Image.open(img_path).convert("RGB").resize((513,513))
        test_img = np.asarray(test_img)
    
        image_transforms = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_in = image_transforms(test_img)
        test_in = Variable(test_in.cuda())
        print(test_in.shape)
        print('00000')
        test_out = model(test_in.unsqueeze(0))
        print('11111')
        _, test_pred = torch.max(test_out, 1)
        test_pred = test_pred.data.cpu().numpy().squeeze().astype(np.uint8)
        test_mask_pred = Image.fromarray(test_pred)
        test_mask_pred.putpalette(cmap)
        test_img = Image.fromarray(test_img, 'RGB')
        img_add = cv2.addWeighted(np.asarray(test_img), 0.3, np.asarray(test_mask_pred.convert('RGB')), 0.7, 0)
        cv2.imshow('Segmentation with mask', img_add)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def main():
    argparser = argparse.ArgumentParser(
        description='Segmentation with mask')
    argparser.add_argument(
        '--folder',
        metavar='FOLDER',
        default='_out',
        help='vedio folder name (default: "_out")')
    args = argparser.parse_args()

    try:
        visualize_segmentation_with_mask(args.folder)
    except KeyboardInterrupt:
        print('done!')

if __name__ == '__main__':
    main()