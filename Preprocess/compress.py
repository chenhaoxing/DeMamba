import subprocess
from glob import glob

def convert_and_compress_video(input_video_path, output_video_path, crf=23, preset='medium'):
    """
    使用ffmpeg转换并压缩视频
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出压缩后视频文件路径
    :param crf: Constant Rate Factor，值越小质量越好，但文件也越大，默认为23是一个平衡点
    :param preset: 预设值，影响压缩速度和文件大小，如'ultrafast', 'fast', 'medium', 'slow', 'veryslow'等，默认'medium'
    """
    # 检查文件是否为GIF
    if input_video_path.lower().endswith('.gif'):
        # 构建ffmpeg命令
        # 首先将GIF转换为MP4
        convert_cmd = f'ffmpeg -i "{input_video_path}" -vf "palettegen" -y palette.png'
        subprocess.run(convert_cmd, shell=True, check=True)
        convert_cmd = f'ffmpeg -i "{input_video_path}" -i palette.png -lavfi "paletteuse" -c:v libx264 -preset {preset} -crf {crf} -c:a copy "{output_video_path}"'
    else:
        # 直接压缩视频
        convert_cmd = f'ffmpeg -i "{input_video_path}" -c:v libx264 -preset {preset} -crf {crf} -c:a copy "{output_video_path}"'
    
    # 执行ffmpeg命令
    try:
        subprocess.run(convert_cmd, shell=True, check=True)
        print(f"视频压缩完成，输出文件：{output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"视频压缩失败：{e}")

video_paths = '/home3/Transformer'
all_dirs = glob(video_paths+'/*')
output_dirs = '/home3/robust/compress/Transformer_'

for path in all_dirs:
    name = path.split('/')[-1]
    out_path = output_dirs + name.replace('.gif', '.mp4') if path.lower().endswith('.gif') else output_dirs + name
    print(name)
    convert_and_compress_video(path, out_path, crf=28)
