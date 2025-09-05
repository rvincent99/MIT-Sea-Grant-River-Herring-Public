# MIT-Sea-Grant-River-Herring-Public
This repository contains code, models, tools, and documentation for a river herring monitoring system that uses underwater video and computer vision. It supports a complete workflow including: video processing and annotation, training custom YOLO-based detection models, tracking fish movements, generating fish counts and applying unbiased count corrections via importance sampling.
 


## ⚙️ Requirement
  * Platforms: Windows, Linux or maxOS
  * NVIDIA GPU (recommended for CV model training and inference).
  * python>=3.9 (tested on 3.12)
  * Pytorch with CUDA if has a CUDA-enabled GPU
  * [Ultralytic YOLO v11](https://github.com/ultralytics/ultralytics)
  * [Supervision](https://supervision.roboflow.com/latest/)
  * huggingface_hub (for downloading test dataset)
  * other depedencies listed in requirements.txt

## 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at [lila.science](https://lila.science/datasets/mit-sea-grant-river-herring/).      
It is also included in the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset/).  



## 🤖 YOLO model training

#### 1. Download test dataset (a small subset for testing)
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


## 🐟 3. Counting Fish
A yolo11 model pretrained using the full dataset is available under `weights/` for river herring detection and counting. The speed of processing each video mostly depending on GPU.


```python
python src/count_fish.py \
    data/raw_video/1_2024-05-07_10_06_48-355.mp4 \ # input video file
    weights/river-herring-yolo11.pt \              # model weight
    outputs/fish_count \                           # output dir for count results
    --class_id 0 \          # Class ID to count
    --save_video \          # include to save annotated video
    --tracker 'botsort.yaml' \  # tracker "bytetrack.yaml"
    --conf_thresh 0.8 \     # detection threshold
    --line_pos 0.6 \        # count line position, left - 0, right - 1
    --move_right "Upstream" \  # migration direction of fish swiming right
    --move_left "Downstream"   # migration direction of fish swiming left
```

**Batch processing**
To process multiple videos at once, list their file paths in `scripts/video_file_list.txt`. Then, run this bash script. The results for each video will be saved in a dedicated directory (named after the video file) within the specified output directory (outdir).

```bash
./scripts/batch_fish_counter.sh  scripts/video_file_list.txt weights/river-herring-yolo11.pt outputs/fish_count

# Arguments:
#   VIDEO_LIST_FILE    Text file containing video file paths (one per line)
#   WEIGHTS           Path to YOLO model weights
#   OUT_DIR           Base output directory for all results

#Options:
#   --python-script PATH   Path to process_video_cli.py (default: src/count_fish.py)
#   --class-id ID         Class ID to count (default: 0)
#   --conf-thresh THRESH  Confidence threshold (default: 0.7)
#   --line-pos POS        Line position 0.0-1.0 (default: 0.5)
#   --imgsz W H          Input image size (default: 480 320)
#   --tracker CONFIG     Tracker config file (default: bytetrack.yaml)
#   --move-right LABEL   Label for rightward movement (default: Up)
#   --move-left LABEL    Label for leftward movement (default: Down)
#   --save-video         Save annotated videos
#   --continue-on-error  Continue processing even if a video fails
#   -h, --help          Show this help message

```






## Useful Links

<details><summary> <b>Expand</b> </summary>

Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

CVAT: https://www.cvat.ai/resources/blog/bounding-boxes

Ultralytics YOLO: https://github.com/ultralytics/ultralytics

Supervision: https://supervision.roboflow.com/develop/notebooks/count-objects-crossing-the-line/

FFmpeg: https://ffmpeg.org/

W&B: https://docs.wandb.ai/guides/integrations/ultralytics/

</details>




