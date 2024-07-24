This is the official code of paper 'DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark'.

## :dart: Todo
- [x] Release source code.
- [ ] Release dataset. (We are uploading the dataset, which may take a few days due to speed limitations.)


## :file_folder: Dataset download
![](figs/tab_fig.jpg)

### Data preparation process
 - Download the original videos.
   
   - Generated videos: all generated videos can download at [https://modelscope.cn/datasets/cccnju/Gen-Video](https://modelscope.cn/datasets/cccnju/Gen-Video).
     
   - Real videos: The data from the MSRVTT dataset is contained within the GenVideo-Val.zip file. We also provided the selected Youku videos in previous link . For Kinetics-400, you will need to download it yourself at [https://github.com/cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).
     
 - Preprocess the video and get the data list csv file.

Statistics of real and generated videos in the GenVideo dataset:
| **Video Source**                                    | **Type** | **Task** | **Time** | **Resolution** | **FPS** | **Length** | **Training Set** | **Testing Set** | **Total Count** |
|-----------------------------------------------------|----------|----------|----------|----------------|---------|------------|------------------|----------------|-----------------|
| Kinetics-400               | Real     | -        | 17.05    | 224-340        | -       | 5-10s      | 260,232          | -              | 1,213,511       |
| Youku-mPLUG~                     | Real     | -        | 23.07    | -              | -       | 10-120s    | 953,279          | -              |                 |
| MSR-VTT                           | Real     | -        | 16.05    | -              | -       | 10-30s     | -                | 10,000         | 10,000          |
| ZeroScope                       | Fake     | T2V      | 23.07    | 1024×576       | 8       | 3s         | 133,169          | -              | 1,048,575       |
| I2VGen-XL                         | Fake     | I2V      | 23.12    | 1280×720       | 8       | 2s         | 61,975           | -              |                 |
| SVD                     | Fake     | I2V      | 23.12    | 1024×576       | 8       | 4s         | 149,026          | -              |                 |
| VideoCrafte          | Fake     | T2V      | 24.01    | 1024×576       | 8       | 2s         | 39,485           | -              |                 |
| Pika                                   | Fake     | T2V&I2V  | 24.02    | 1088×640       | 24      | 3s         | 98,377           | -              |                 |
| DynamiCrafter         | Fake     | I2V      | 24.03    | 1024×576       | 8       | 3s         | 46,205          | -              |                 |
| SD                             | Fake     | T2V&I2V  | 23-24    | 512-1024       | 8       | 2-6s       | 200,720          | -              |                 |
| SEINE                         | Fake     | I2V      | 24.04    | 1024×576       | 8       | 2-4s       | 24,737            | -              |                 |
| Latte                           | Fake     | T2V      | 24.03    | 512×512        | 8       | 2s         | 149,979          | -              |                 |
| OpenSora                           | Fake     | T2V      | 24.03    | 512×512        | 8       | 2s         | 177,410          | -              |                 |
| ModelScope               | Fake     | T2V      | 23.03    | 256×256        | 8       | 4s         | -                | 700            | 8,588           |
| MorphStudio                     | Fake     | T2V      | 23.08    | 1280×720       | 8       | 2s         | -                | 700            |                 |
| MoonValley                       | Fake     | T2V      | 24.01    | 1024×576       | 16      | 3s         | -                | 626            |                 |
| HotShot                           | Fake     | T2V      | 23.10    | 672×384        | 8       | 1s         | -                | 700            |                 |
| Show_1                       | Fake     | T2V      | 23.10    | 576×320        | 8       | 4s         | -                | 700            |                 |
| Gen2                     | Fake     | I2V&T2V  | 23.09    | 896×512        | 24      | 4s         | -                | 1,380          |                 |
| Crafter               | Fake     | T2V      | 23.04    | 256×256        | 8       | 4s         | -                | 1,400          |                 |
| Lavie                                 | Fake     | T2V      | 23.09    | 1280×2048      | 8       | 2s         | -                | 1,400          |                 |
| Sora                                 | Fake     | T2V      | 24.02    | -              | -       | -60s       | -                | 56             |                 |
| WildScrape                                          | Fake     | T2V&I2V  | 24       | 512-1024       | 8-16    | 2-6s       | -                | 926            |                 |
| **Total Count**                                     | -        | -        | -        | -              | -       | -          | 2,294,594       | 19,588         | 2,314,182       |

## :snake: Detail Mamba (DeMamba)

![](figs/logo.png)
<p align="center"><em>In memory of Kobe Bryant (generated by GPT-4o)</em></p>

> "Determination wins games, but Detail wins championships." — *Kobe Bryant, in his Show Detail, 2018*

![](figs/VFOD.png)
<p align="center"><em>The overall framework of our Detail Mamba (DeMamba)</em></p>


## :chart_with_upwards_trend: Leaderboard 

### Many-to-many generalization task
| Model         | Detection Level | Metric | Sora  | Morph | Gen2  | HotShot | Lavie | Show-1 | Moon  | Crafter | Model Scope | Wild Scrape | Real  | Avg.   |
|---------------|-----------------|--------|-------|-------|-------|---------|-------|--------|-------|---------|-------------|-------------|-------|--------|
| F3Net         | Image           | ACC    | 83.93 | 99.71 | 98.62 | 77.57   | 57.00 | 36.57  | 99.52 | 99.71   | 89.43       | 76.78       | 99.14 | 83.45  |
|               |                 | AP     | 68.27 | 99.89 | 99.67 | 89.35   | 85.24 | 63.17  | 99.58 | 99.89   | 93.80       | 88.41       | -     | 88.73  |
| NPR           | Image           | ACC    | 91.07 | 99.57 | 99.49 | 24.29   | 89.64 | 57.71  | 97.12 | 99.86   | 94.29       | 87.80       | 97.46 | 85.30  |
|               |                 | AP     | 67.17 | 99.14 | 99.20 | 22.76   | 93.91 | 61.76  | 96.33 | 99.72   | 94.15       | 90.40       | -     | 82.45  |
| STIL          | Video           | ACC    | 78.57 | 98.14 | 98.04 | 76.00   | 61.79 | 53.29  | 99.36 | 97.36   | 94.57       | 65.01       | 98.72 | 83.71  |
|               |                 | AP     | 57.21 | 99.08 | 99.32 | 86.19   | 82.24 | 70.43  | 99.25 | 98.96   | 97.18       | 81.32       | -     | 87.12  |
| VideoMAE-B    | Video           | ACC    | 67.86 | 96.00 | 98.41 | 96.14   | 77.14 | 80.43  | 97.44 | 96.93   | 96.29       | 68.36       | 99.71 | 88.61  |
|               |                 | AP     | 66.49 | 98.85 | 99.77 | 99.27   | 96.55 | 95.31  | 99.49 | 99.69   | 99.27       | 90.74       | -     | 94.54  |
| CLIP-B-PT     | Image           | ACC    | 85.71 | 82.43 | 90.36 | 71.00   | 79.29 | 75.43  | 89.62 | 86.29   | 82.14       | 75.16       | 57.22 | 79.67  |
|               |                 | AP     | 6.78  | 43.56 | 70.88 | 29.97   | 52.97 | 35.36  | 55.52 | 66.03   | 44.23       | 42.99       | -     | 44.83  |
| DeMamba-CLIP-PT| Video          | ACC    | 58.93 | 96.43 | 93.12 | 68.00   | 69.36 | 69.00  | 89.14 | 91.86   | 96.14       | 56.59       | 98.06 | 80.60 |
|               |                 | AP     | 25.87 | 95.14 | 96.23 | 73.43   | 83.31 | 75.49  | 90.17 | 95.06   | 95.05       | 69.95       | -     | 79.97 |
| CLIP-B-FT     | Image           | ACC    | 94.64 | 99.86 | 91.38 | 77.29   | 88.14 | 86.00  | 99.68 | 99.79   | 84.29       | 84.67       | 97.38 | 91.19  |
|               |                 | AP     | 80.67 | 99.67 | 95.24 | 82.20   | 93.48 | 88.62  | 99.55 | 99.79   | 86.93       | 89.08       | -     | 91.52  |
| DeMamba-CLIP-FT| Video          | ACC    | 95.71 | 100.00| 98.70 | 69.14   | 92.43 | 93.29  | 100.00| 100.00  | 83.57       | 82.94       | 99.44 | $92.29 |
|               |                 | AP     | 85.50 | 100.00| 99.59 | 76.15   | 96.78 | 96.99  | 99.97 | 100.00  | 89.80       | 89.72       | -     | $93.45 |
| XCLIP-B-PT    | Video           | ACC    | 81.34 | 82.15 | 83.35 | 80.98   | 81.82 | 81.55  | 82.14 | 82.98   | 81.93       | 81.10       | 81.37 | 81.88  |
|               |                 | AP     | 16.39 | 72.16 | 87.77 | 39.86   | 65.57 | 54.26  | 75.23 | 84.80   | 61.60       | 55.28       | -     | 61.29  |
| DeMamba-XCLIP-PT| Video         | ACC    | 66.07 | 95.86 | 94.64 | 77.86   | 75.36 | 80.29  | 90.89 | 92.50   | 96.00       | 66.41       | 95.12 | $84.64|
|               |                 | AP     | 18.26 | 93.50 | 94.72 | 69.94   | 78.08 | 71.50  | 83.95 | 92.23   | 93.54       | 68.10       | -     | $76.38 |
| XCLIP-B-FT    | Video           | ACC    | 82.14 | 99.57 | 93.62 | 61.29   | 79.36 | 69.71  | 97.92 | 99.79   | 77.14       | 83.59       | 98.14 | 85.66  |
|               |                 | AP     | 64.42 | 99.73 | 96.78 | 70.98   | 90.35 | 77.28  | 97.34 | 99.84   | 82.01       | 88.97       | -     | 86.77  |
| DeMamba-XCLIP-FT| Video         | ACC    | 98.21 | 100.00| 99.86 | 65.43   | 94.86 | 98.86  | 100.00| 100.00  | 92.86       | 89.09       | 99.42 | $**94.42**|
|               |                 | AP     | 93.32 | 100.00| 99.97 | 85.55   | 98.97 | 99.60  | 99.98 | 100.00  | 97.77       | 95.75       | -     | $**97.10**|



### One-to-many generalization task

| Training | Model | Detection | Metric | Sora | Morph | Gen2 | HotShot | Lavie | Show-1 | Moon | Crafter | Model | Wild | Real | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pika | NPR | Image | ACC | 55.36 | 77.57 | 71.88 | 4.86 | 7.21 | 4.29 | 86.26 | 60.29 | 71.43 | 31.53 | 99.52 | 51.83 |
| | | | AP | 45.74 | 91.55 | 92.71 | 21.80 | 44.32 | 22.74 | 95.04 | 90.03 | 84.91 | 60.88 | - | 64.97 |
| | STIL | Image | ACC | 75.00 | 79.43 | 94.49 | 57.86 | 53.14 | 64.14 | 97.12 | 85.29 | 69.43 | 62.42 | 92.43 | 75.52 |
| | | | AP | 22.35 | 71.62 | 93.19 | 40.61 | 53.24 | 47.73 | 94.94 | 85.82 | 58.99 | 61.91 | - | 63.04 |
| | XCLIP-B-FT | Video | ACC | 67.86 | 91.29 | 96.23 | 12.00 | 22.36 | 9.14 | 99.84 | 83.43 | 75.57 | 51.84 | 99.64 | 64.47 |
| | | | AP | 71.08 | 97.53 | 99.44 | 44.68 | 72.69 | 38.37 | 99.96 | 97.32 | 88.00 | 74.00 | - | 78.31 |
| | DeMamba | Video | ACC | 92.86 | 97.29 | 98.48 | 38.29 | 53.50 | 41.43 | 99.84 | 94.07 | 77.29 | 64.15 | 98.65 | **77.80** |
| | -XCLIP-FT | | AP | 77.75 | 98.42 | 99.16 | 52.97 | 76.72 | 56.24 | 99.80 | 97.91 | 82.83 | 74.81 | - | **81.66** |
| SEINE | NPR | Image | ACC | 46.43 | 78.57 | 63.70 | 21.86 | 7.00 | 3.29 | 92.97 | 89.29 | 33.86 | 24.84 | 99.70 | 51.05 |
| | | | AP | 36.30 | 92.63 | 85.02 | 52.68 | 25.69 | 11.05 | 97.80 | 97.78 | 64.64 | 47.48 | - | 61.11 |
| | STIL | Video | ACC | 71.43 | 80.43 | 88.48 | 67.71 | 54.57 | 55.71 | 93.93 | 89.57 | 72.00 | 50.11 | 92.27 | 74.20 |
| | | | AP | 23.89 | 71.01 | 88.18 | 52.17 | 54.49 | 41.23 | 84.73 | 87.38 | 58.72 | 46.51 | - | 60.83 |
| | XCLIP-B-FT | Video | ACC | 85.71 | 95.43 | 76.23 | 65.86 | 35.93 | 37.00 | 99.68 | 99.00 | 75.57 | 49.78 | 99.80 | 74.54 |
| | | | AP | 85.89 | 97.97 | 94.40 | 92.81 | 81.68 | 77.68 | 98.48 | 98.91 | 92.27 | 67.91 | - | 88.80 |
| | DeMamba | Video | ACC | 94.64 | 98.43 | 92.17 | 82.43 | 52.29 | 54.00 | 99.52 | 99.14 | 79.29 | 57.88 | 98.99 | **82.61** |
| | -XCLIP-FT | | AP | 83.74 | 99.01 | 97.66 | 90.82 | 84.11 | 73.30 | 99.72 | 99.73 | 89.72 | 76.45 | - | **89.43** |
| OpenSora | NPR | Image | ACC | 55.36 | 76.29 | 55.51 | 58.57 | 76.50 | 22.43 | 74.92 | 83.07 | 29.86 | 60.37 | 95.95 | 62.62 |
| | | | AP | 25.65 | 75.24 | 65.02 | 55.12 | 82.42 | 20.75 | 72.65 | 86.84 | 28.13 | 64.50 | - | 57.63 |
| | STIL | Video | ACC | 32.14 | 45.43 | 56.45 | 35.14 | 45.07 | 34.57 | 57.83 | 63.14 | 19.86 | 43.95 | 98.13 | 48.33 |
| | | | AP | 6.94 | 55.62 | 75.99 | 43.55 | 68.06 | 44.01 | 63.84 | 80.59 | 29.39 | 57.58 | - | 52.56 |
| | XCLIP-B-FT | Video | ACC | 67.86 | 75.86 | 67.46 | 70.86 | 73.14 | 43.57 | 79.87 | 86.29 | 33.43 | 63.17 | 98.10 | 69.06 |
| | | | AP | 48.28 | 81.39 | 81.84 | 77.38 | 86.08 | 51.87 | 83.41 | 93.18 | 39.27 | 72.74 | - | 71.54 |
| | DeMamba | Video | ACC | 55.36 | 87.43 | 81.30 | 73.14 | 85.21 | 73.14 | 89.62 | 90.07 | 44.86 | 58.10 | 97.30 | **75.95** |
| | -XCLIP-FT| | AP |  25.89 | 86.63  |  87.27  |  74.38 | 91.12 | 76.01 | 86.41 | 93.83 | 48.74 |  67.92 | - | **73.82** ｜

## :space_invader: Citing GenVideo&DeMamba
If you use GenVideo or DeMamba in your research or use the codebase here, please use the following BibTeX entry.

```BibTeX
@article{DeMamba,
      title={DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark},
      author={Haoxing Chen and Yan Hong and Zizheng Huang and Zhuoer Xu and Zhangxuan Gu and Yaohui Li and Jun Lan and Huijia Zhu and Jianfu Zhang and Weiqiang Wang and Huaxiong Li},
      journal={arXiv preprint arXiv:2405.19707},
      year={2024}
}
```

## Acknowledgement
Many thanks to the nice work of [STIL](https://github.com/wizyoung/STIL-DeepFake-Video-Detection), [CLIP](https://github.com/openai/CLIP), [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP), [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/tree/main) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE-Action-Detection). 

## :email: Contact
If you have any questions, feel free to contact us: hx.chen@hotmail.com.



