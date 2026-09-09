import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import math
import functools
import json
import copy
import numpy as np
from .transforms import *
import pickle as pkl
import shutil
from utils import UCF_DATA_ROOT

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        #image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i)) if ucf101
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i)) 
        #print(image_path)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class attack_ucf101(data.Dataset):
    def __init__(self, setting_path, spatial_transform=None, temporal_transform=None,get_loader=get_default_video_loader):
        setting = setting_path
        self.clips = self._make_dataset(setting)    
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        print ('length', len(self.clips))
        
    def __getitem__(self, index):
        directory, duration, target = self.clips[index]
        frame_indices = list(range(1, duration + 1,4))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(directory, frame_indices)   
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, target,index


    def _make_dataset(self, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        #'/home/pingchenhao/sthv2/'
        target_root ='/data/somethingv2/frame/'
        clips = []
        with open(setting, 'r') as f:
            lines = f.readlines()
            for row in lines:
                row=row.split()
                name=row[0]
                clip_path = os.path.join('/data/somethingv2/frame', name)
                duration = int(row[1])
                target = int(row[2])
                item = (clip_path, duration, target)
                name = os.path.basename(clip_path)
                target_dir = os.path.join(target_root, name)
                # os.makedirs(target_dir, exist_ok=True)
                clips.append(item)
                # for i in range(1, 32):
                #     image_name = f'{i:06d}.jpg'  # 格式化为六位数字，如000001.jpg
                #     src_image_path = os.path.join(clip_path, image_name)
                #     dest_image_path = os.path.join(target_dir, image_name)

                #     if os.path.exists(src_image_path):
                #         shutil.copy(src_image_path, dest_image_path)
                #         print(f'Copied: {src_image_path} -> {dest_image_path}')
                #     else:
                #         print(f'Warning: {src_image_path} not found.')
        return clips
    
    def __len__(self):
        return len(self.clips)

def test_transform():
    input_size = 224
    scale_ratios = '1.0, 0.8'
    scale_ratios = [float(i) for i in scale_ratios.split(',')]
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    norm_method = Normalize(default_mean, default_std)
    spatial_transform = spatial_Compose([
       Scale(int(input_size / 1.0)),
        CornerCrop(input_size, 'c'),
        #MultiScaleCornerCrop([0.8, 0.9, 1.0],input_size,Image.BILINEAR,'c'),
        ToTensor(), norm_method
        ])
    temporal_transform = LoopPadding(16)
    return spatial_transform, temporal_transform

def get_dataset(setting_path, test_batch_size, loader=True):
    test_spa_trans, test_temp_trans = test_transform()
    test_dataset = attack_ucf101(setting_path, spatial_transform=test_spa_trans, temporal_transform=test_temp_trans)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=9, drop_last=True,pin_memory=True)
    #print(len(test_dataset))
    return val_loader