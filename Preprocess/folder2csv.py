import os
import csv
import pandas as pd
from pandas import Series, DataFrame
from glob import glob
import os

def count_images_in_folder(folder_path):
    image_count = 0
    image_names = []  
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            image_count += 1
            image_names.append(int(file_name.split('.')[0]))
    image_names.sort()
    return image_count, image_names


folder_path = './SD_frames'
all_dirs = []

for root, dirs, files in os.walk(folder_path):
    for dir in dirs:
        all_dirs.append(os.path.join(root, dir))

label = list()
save_path = list()
frame_counts = list()
frame_seq_counts = list()
content_paths = list()
chinese_labels = list()


for video_path in all_dirs:
    frame_paths = glob(video_path + '/*')
    temp_frame_count, temp_frame_seqs = count_images_in_folder(video_path)
    if temp_frame_count == 0:
        continue

    for frame in frame_paths:
        content_path = frame.split('/')[1:-1]
        content_path = '/'.join(content_path)
        # input your own path
        content_path = '/home/AIGC_Video_Det/SD/' + content_path
      
        frame_path = frame.split('/')[1:]
        frame_path = '/'.join(frame_path)
        frame_path = '/home/AIGC_Video_Det/SD/' + frame_path

        print(content_path, frame_path)
        label.append(str(1))
        frame_counts.append(int(temp_frame_count))
        frame_seq_counts.append(temp_frame_seqs)
        save_path.append(frame_path)
        content_paths.append(content_path)
        chinese_labels.append('AIGC视频')
        # chinese_labels.append('真实视频')
        break

dic={
    'content_path': Series(data=content_paths),
    'image_path': Series(data=save_path),
    'type_id': Series(data=chinese_labels),
    'label': Series(data=label),
    'frame_len': Series(data=frame_counts),
    'frame_seq': Series(data=frame_seq_counts)
}

print(dic)
pd.DataFrame(dic).to_csv('SD.csv', encoding='utf-8', index=False)


