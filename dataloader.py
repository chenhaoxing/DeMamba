import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import albumentations
import random
import os
import numpy as np
import cv2
import math
import warnings


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]

    return cropped_image

class Ours_Dataset_train(Dataset):
    def __init__(self, index_list=None, df=None):
        self.index_list = index_list
        self.df = df
        self.positive_indices = df[df['label'] == 1].index.tolist()
        self.negative_indices = df[df['label'] == 0].index.tolist()
        self.balanced_indices = []
        self.resample()

    def resample(self):
        # Ensure each epoch uses a balanced dataset
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        self.balanced_indices.clear()
        self.balanced_indices.extend(random.sample(self.positive_indices, min_samples))
        self.balanced_indices.extend(random.sample(self.negative_indices, min_samples))
        random.shuffle(self.balanced_indices)  # Shuffle to mix positive and negative samples

    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        row = self.df.iloc[real_idx]
        video_id = row['content_path']
        label = row['label']
        frame_list = eval(row['frame_seq'])
        label_onehot = [0]*2
        select_frame_nums = 8

        aug_list  = [
                    albumentations.Resize(224, 224)
                    ]

        if random.random() < 0.5:
            aug_list.append(albumentations.HorizontalFlip(p=1.0))
        if random.random() < 0.5:
            quality_score = random.randint(50, 100)
            aug_list.append(albumentations.ImageCompression(quality_lower=quality_score, quality_upper=quality_score))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussNoise(p=1.0))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
        if random.random() < 0.001:
            aug_list.append(albumentations.ToGray(p=1.0))
            
        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)

        if len(frame_list) >= select_frame_nums:
            start_frame = random.randint(0, len(frame_list)-select_frame_nums)
            select_frames = frame_list[start_frame:start_frame+select_frame_nums]
            frames = []
            for x in frame_list[start_frame:start_frame+select_frame_nums]:
                while True:
                    try:
                        temp_image_path = video_id+'/'+str(x)+'.jpg'
                        image = download_oss_file('GenVideo/'+ temp_image_path)
                        if video_id.startswith("real/youku"):
                            image = crop_center_by_percentage(image, 0.15)
                        break
                    except Exception as e:
                        if x+1 < len(frame_list):
                            x = x + 1
                        elif x - 1 >=0 :
                            x = x - 1
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])
        else:
            pad_num = select_frame_nums-len(frame_list)
            frames = []
            for x in frame_list:
                temp_image_path = video_id+'/'+str(x)+'.jpg'
                image = download_oss_file('GenVideo/'+temp_image_path)
                if video_id.startswith("real/youku"):
                    image = crop_center_by_percentage(image, 0.15)
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])    
            for i in range(pad_num):
                frames.append(np.zeros((224,224,3)).transpose(2,0,1)[np.newaxis,:])
        
        label_onehot[int(label)] = 1
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis,:])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])

        return self.index_list[idx], frames, label_onehot, binary_label

    def __len__(self):
        return len(self.balanced_indices)


class Ours_Dataset_val(data.Dataset):
    def __init__(self, cfg, index_list=None, df=None):
        self.index_list = index_list
        self.cfg = cfg
        self.df = df
        self.frame_dir = df['image_path'].tolist()

    def __getitem__(self, idx):
        aug_list  = [
                    albumentations.Resize(224, 224),
                    ]
        
        if self.cfg['task'] == 'JPEG_Compress_Attack':
            aug_list.append(albumentations.JpegCompression(quality_lower=35, quality_upper=35,p=1.0))
        if self.cfg['task'] == 'FLIP_Attack':
            if random.random() < 0.5:
                aug_list.append(albumentations.HorizontalFlip(p=1.0))
            else:
                aug_list.append(albumentations.VerticalFlip(p=1.0))
        if self.cfg['task'] == 'CROP_Attack':
            random_crop_x = random.randint(0, 16)  
            random_crop_y = random.randint(0, 16)  
            crop_width = random.randint(160, 208) 
            crop_height = random.randint(160, 208)
            aug_list.append(albumentations.Crop(x_min=random_crop_x, y_min=random_crop_y, x_max=random_crop_x+crop_width, y_max=random_crop_y+crop_height))
            aug_list.append(albumentations.Resize(224, 224))

        if self.cfg['task'] == 'Color_Attack':
            index = random.choice([i for i in range(4)])
            dicts = {0:[0.5,0,0,0],1:[0,0.5,0,0],2:[0,0,0.5,0],3:[0,0,0,0.5]}
            brightness,contrast,saturation,hue = dicts[index]
            aug_list.append(albumentations.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

        if self.cfg['task'] == 'Gaussian_Attack':     
            aug_list.append(albumentations.GaussianBlur(blur_limit=(7, 7), p=1.0))

        aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)


        df_v = self.df.loc[self.index_list[idx]]
        video_id = df_v['content_path']
        activity_id = df_v['activity_id']
        label = df_v['label']
        label_onehot = [0]*2
        frame_list = eval(df_v['frame_seq'])

        select_frame_nums = 8

        if len(frame_list) >= select_frame_nums:
            start_frame = random.randint(0, len(frame_list)-select_frame_nums)
            select_frames = frame_list[start_frame:start_frame+select_frame_nums]
            frames = []
            for x in frame_list[start_frame:start_frame+select_frame_nums]:
                while True:
                    try:
                        temp_image_path = video_id+'/'+str(x)+'.jpg'
                        image = download_oss_file('GenVideo/'+ temp_image_path)
                        image = crop_center_by_percentage(image, 0.1)
                        break
                    except Exception as e:
                        if x+1 < len(frame_list):
                            x = x + 1
                        elif x - 1 >=0 :
                            x = x - 1
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])
        else:
            pad_num = select_frame_nums-len(frame_list)
            frames = []
            for x in frame_list:
                temp_image_path = video_id+'/'+str(x)+'.jpg'
                image = download_oss_file('GenVideo/'+temp_image_path)
                image = crop_center_by_percentage(image, 0.1)
                augmented = trans(image=image)
                image = augmented["image"]
                frames.append(image.transpose(2,0,1)[np.newaxis,:])    
            for i in range(pad_num):
                frames.append(np.zeros((224,224,3)).transpose(2,0,1)[np.newaxis,:])

        label_onehot[int(label)] = 1
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis,:])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])
        return self.index_list[idx], frames, label_onehot, binary_label, video_id

    
    def __len__(self):
        return len(self.index_list)



def generate_dataset_loader(cfg):
    df_train = pd.read_csv('GenVideo/datasets/train.csv')

    if cfg['task'] == 'normal':
        df_val = pd.read_csv('GenVideo/datasets/val_id.csv')
    elif cfg['task'] == 'robust_compress':
        df_val = pd.read_csv('GenVideo/datasets/com_28.csv')
    elif cfg['task'] == 'Image_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/imgwater.csv')
    elif cfg['task'] == 'Text_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/textwater.csv')
    elif cfg['task'] == 'one2many':
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')
        if cfg['train_sub_set'] == 'pika':
            prefixes = ["fake/pika", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'SEINE':
            prefixes = ["fake/SEINE", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'OpenSora':
            prefixes = ["fake/OpenSora", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'Latte':
            prefixes = ["fake/Latte", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
    else:
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    
    index_val = df_val.index.tolist()
    index_val = index_val[:]

    val_dataset = Ours_Dataset_val(cfg, index_val, df_val)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True, drop_last=False
        )

    index_train = df_train.index.tolist()
    index_train = index_train[:]
    train_dataset = Ours_Dataset_train(index_train, df_train)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True
        )

    print("******* Training Video IDs", str(len(index_train))," Training Batch size ", str(cfg['train_batch_size'])," *******")
    print("******* Testing Video IDs", str(len(index_val)), " Testing Batch size ", str(cfg['val_batch_size'])," *******")

    return train_loader, val_loader


