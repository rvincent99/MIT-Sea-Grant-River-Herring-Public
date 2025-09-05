# MIT-Sea-Grant-River-Herring-Public
This repository contains code, models, tools, and documentation for a river herring monitoring system that uses underwater video and computer vision. It supports a complete workflow including: video processing and annotation, training custom YOLO-based detection models, tracking fish movements, generating fish counts and applying unbiased count corrections via importance sampling.
 


## ⚙️ Requirement
  * Platforms: Windows, Linux or maxOS
  * NVIDIA GPU (recommended for CV model training and inference).
  * python>=3.9 (tested on 3.12)
  * Pytorch with CUDA if has a CUDA-enabled GPU
  * [Ultralytic YOLO v11](https://github.com/ultralytics/ultralytics)
  * [Supervision](https://supervision.roboflow.com/latest/)
  * huggingface_hub (for test dataset)
  * other depedencies listed in requirements.txt

## 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at:  https://lila.science/datasets/mit-sea-grant-river-herring/   
It is also included in the Community Fish Detection Dataset (https://lila.science/datasets/community-fish-detection-dataset/).  

```

torch==2.8.0+cu128
torchvision==0.23.0+cu128
ultralytics==8.3.193
supervision==0.26.1
pip install -r requirements.txt

```

## YOLO model training

#### 1. Download test dataset
```python
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(repo_id="zhongqic/Fisheye-example", allow_patterns=["*.tar.gz", "data.yaml"], repo_type="dataset", local_dir="data/test_train")

```

```bash
# Unzip downloaded files
tar  -xzf data/test_train/train.tar.gz -C .\data\test_train
tar  -xzf data/test_train/val.tar.gz -C .\data\test_train
tar  -xzf data/test_train/test.tar.gz -C .\data\test_train
```

#### 2. YOLO model training
Ultralytics YOLO training is nicely packed and very easy to run.  Python scripts here can also be found in `src/train_yolo11.py`.  
```python

from ultralytics import YOLO

weights_path = "weights/yolo11l.pt"
yaml_file    = "data/test_train/data.yaml"

# Load a model
model = YOLO(weights_path) 
# Train the model
results = model.train(data=yaml_file, epochs=2, batch = 8)
# Validate on the test set
model.val(data  = yaml_file, split = "test")
```




## Quick start - Run pre-trained model  
(1) Setup environment: `conda env create -f environment.yml && conda activate fish-ai`  
(2) Download weights  to `/models`  
(3) Put video files in `data/raw/videos/` and list them in `examples/video_list.txt`  
(4) Run: `bash scripts/inference.sh examples/video_list.txt`  



## Useful Links

<details><summary> <b>Expand</b> </summary>

Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

CVAT: https://www.cvat.ai/resources/blog/bounding-boxes

Ultralytics YOLO: https://github.com/ultralytics/ultralytics

Supervision: https://supervision.roboflow.com/develop/notebooks/count-objects-crossing-the-line/

FFmpeg: https://ffmpeg.org/

W&B: https://docs.wandb.ai/guides/integrations/ultralytics/

</details>




