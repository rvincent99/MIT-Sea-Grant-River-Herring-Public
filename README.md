# MIT-Sea-Grant-River-Herring-Public
This repository contains code, models, tools, and documentation for a river herring monitoring system that uses underwater video and computer vision. It supports a complete workflow including: video processing and annotation, training custom YOLO-based detection models, tracking fish movements, generating fish counts and applying unbiased count corrections via importance sampling.
 


## ⚙️ Requirement
  * Platforms: Windows, Linux or maxOS
  * NVIDIA GPU (recommended for CV model training and inference).
  * python>=3.8 (tested on 3.12)
  * [Ultralytic YOLO v11](https://github.com/ultralytics/ultralytics)
  * [Supervision](https://supervision.roboflow.com/latest/)
  * other depedencies listed in requirements.txt

## 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at:  https://lila.science/datasets/mit-sea-grant-river-herring/   
It is also included in the Community Fish Detection Dataset (https://lila.science/datasets/community-fish-detection-dataset/).  


## Quick start - Run pre-trained model  
(1) Setup environment: `conda env create -f environment.yml && conda activate fish-ai`  
(2) Download weights  to `/models`  
(3) Put video files in `data/raw/videos/` and list them in `examples/video_list.txt`  
(4) Run: `bash scripts/inference.sh examples/video_list.txt`  


## Quick start - Train new model  
(1) Annotate in CVAT.ai → export **COCO**  
(2) Organize the images and labels into train/, val/, and test/ folders under `data/training`.  
(3) YOLO model training  
(4) Inference with the `best.pt`. Run: `bash scripts/inference.sh examples/video_list.txt`  



## Useful Links

<details><summary> <b>Expand</b> </summary>

Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

CVAT: https://www.cvat.ai/resources/blog/bounding-boxes

Ultralytics YOLO: https://github.com/ultralytics/ultralytics

Supervision: https://supervision.roboflow.com/develop/notebooks/count-objects-crossing-the-line/

FFmpeg: https://ffmpeg.org/

W&B: https://docs.wandb.ai/guides/integrations/ultralytics/

</details>




