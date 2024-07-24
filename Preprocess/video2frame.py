import os
from glob import glob
from moviepy.editor import VideoFileClip
import multiprocessing
import cv2, math

def get_video_length(file_path):
    video = VideoFileClip(file_path)
    return video.duration

def process_video(video_path):
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[:-1]
    video_name = '.'.join(video_name)

    path = video_path.split('/')[1:-1]
    path = '/'.join(path)

    image_path = './SD_frames/'+path+'/'+ video_name+'/'
    print(video_name)
    if os.path.exists(image_path):
        print("路径存在")
    else:
        print(video_name, "路径不存在")
        try:
            try:
                video_length = get_video_length(video_path)
                print(video_name, f"视频长度为：{video_length} 秒")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                if video_length >= 4 :
                    inter_val = 2
                    os.system(f"cd {image_path} | ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}%d.jpg")
                else:
                    inter_val = math.ceil(8 / video_length)
                    os.system(f"cd {image_path} | ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}%d.jpg")
 
            except Exception as e:
                print("发生异常：", str(e))
        except:
            print("Skip")

if __name__ == '__main__':
    print("Getting frames!!")
    video_paths = './SD'
    all_dirs = []
    all_dirs = glob(video_paths+'/*')
    
    print(all_dirs)
    
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_video, all_dirs)
    pool.close()
    pool.join()



